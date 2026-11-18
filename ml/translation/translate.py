"""Reproducible translation pipeline for Hindi (hi) and Bengali (bn) climate stance data.

Implements:
- Deterministic half/half allocation between Hindi and Bengali using a fixed seed
- JSON translation cache keyed by SHA-256(text + target_lang)
- Memory-safe batch processing (greedy decoding num_beams=1, small batches, gc cleanup)
- Resume capability from partial outputs & incremental cache saving
- Script-based language identification check (Devanagari for hi, Bengali for bn)
- Backends:
  * hf       : Local Hugging Face MarianMT batch translation
               (Helsinki-NLP/opus-mt-en-hi and shhossain/opus-mt-en-to-bn)
  * mymemory : MyMemoryTranslator API
  * google   : GoogleTranslator API
  * auto     : MyMemory -> Google fallback
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.preprocessing.cleaner import detect_script_language
from ml.utils import LOGS_DIR, PROCESSED_DIR, TRANSLATED_DIR, ensure_directories, load_config

CACHE_PATH = TRANSLATED_DIR / "translation_cache.json"
LOG_PATH = LOGS_DIR / "translation.log"
_CACHE_LOCK = threading.Lock()

HF_NMT_MODELS = {
    "hi": "Helsinki-NLP/opus-mt-en-hi",
    "bn": "shhossain/opus-mt-en-to-bn",
}


def _setup_logger() -> logging.Logger:
    ensure_directories()
    logger = logging.getLogger("indic_climate.translate")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(fh)
    return logger


def _cache_key(text: str, target_lang: str) -> str:
    return hashlib.sha256(f"{target_lang}::{text}".encode("utf-8")).hexdigest()


def load_cache(cache_path: Path = CACHE_PATH) -> dict[str, str]:
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_cache(cache: dict[str, str], cache_path: Path = CACHE_PATH) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with _CACHE_LOCK:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)


def _make_api_translator(target_lang: str, backend: str) -> Any:
    if backend == "google":
        from deep_translator import GoogleTranslator

        return GoogleTranslator(source="en", target=target_lang)
    from deep_translator import MyMemoryTranslator

    mymem_target = "hi-IN" if target_lang == "hi" else "bn-IN"
    return MyMemoryTranslator(source="en-GB", target=mymem_target)


def translate_with_retry(
    text: str,
    target_lang: str,
    translator: Any = None,
    backend: str = "auto",
    max_retries: int = 2,
    backoff_seconds: float = 0.6,
) -> tuple[str | None, str | None]:
    """Translate a single text with retries and script verification."""
    query_text = text[:480].rsplit(" ", 1)[0] if len(text) > 480 else text
    last_error: str | None = None

    backends = (
        ["custom"]
        if translator is not None
        else (["mymemory", "google"] if backend == "auto" else [backend])
    )

    for b_name in backends:
        active_translator = translator if b_name == "custom" else _make_api_translator(target_lang, b_name)
        for attempt in range(1, max_retries + 1):
            try:
                translated = active_translator.translate(query_text)
                if not translated or not isinstance(translated, str) or not translated.strip():
                    raise ValueError("Empty translation returned")

                expected_lang = "Hindi" if target_lang == "hi" else "Bengali"
                detected_lang, _ = detect_script_language(translated)
                if detected_lang != expected_lang:
                    raise ValueError(
                        f"Language ID check failed: expected {expected_lang}, detected {detected_lang}"
                    )
                return translated.strip(), None
            except Exception as exc:
                last_error = f"[{b_name}] {exc}"
                if attempt < max_retries:
                    time.sleep(backoff_seconds * (2 ** (attempt - 1)))
    return None, last_error


def translate_batch_hf(
    texts: list[str],
    target_lang: str,
    batch_size: int = 2,
    num_beams: int = 1,
    on_batch_done: Any = None,
) -> list[tuple[str | None, str | None]]:
    """Run memory-safe local Hugging Face MarianMT batch translation."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    torch.set_num_threads(2)
    model_id = HF_NMT_MODELS[target_lang]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading local NMT model '{model_id}' on {device} (batch_size={batch_size}, num_beams={num_beams})...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, low_cpu_mem_usage=True)
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.max_length = None
    model.to(device)
    model.eval()

    expected_lang = "Hindi" if target_lang == "hi" else "Bengali"
    outputs: list[tuple[str | None, str | None]] = []

    for start in range(0, len(texts), batch_size):
        chunk = [t[:280] for t in texts[start : start + batch_size]]
        batch_out: list[tuple[str | None, str | None]] = []
        try:
            enc = tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=72,
            ).to(device)
            with torch.inference_mode():
                gen = model.generate(
                    **enc,
                    max_new_tokens=72,
                    num_beams=num_beams,
                    do_sample=False,
                )
            decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
            for txt in decoded:
                cleaned = txt.strip()
                det, _ = detect_script_language(cleaned)
                if det == expected_lang and cleaned:
                    batch_out.append((cleaned, None))
                else:
                    batch_out.append((None, f"Script verification failed: detected={det}"))
            del enc, gen
        except Exception as exc:
            for _ in chunk:
                batch_out.append((None, f"[hf] {exc}"))

        outputs.extend(batch_out)
        if on_batch_done is not None:
            on_batch_done(start, batch_out)

        done = min(len(texts), start + len(chunk))
        if done % max(batch_size * 5, 10) == 0 or done == len(texts):
            print(f"  [hf:{target_lang}] Translated {done}/{len(texts)}")
        gc.collect()

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return outputs


def assign_translation_targets(
    df: pd.DataFrame,
    seed: int = 42,
    group_col: str = "group_id",
) -> pd.DataFrame:
    """Deterministically assign half of unique groups to Hindi ('hi') and half to Bengali ('bn')."""
    out = df.copy()
    if group_col not in out.columns:
        out[group_col] = out["id"] if "id" in out.columns else np.arange(len(out))

    unique_groups = sorted(out[group_col].unique().tolist())
    rng = np.random.default_rng(seed)
    shuffled = np.array(unique_groups, dtype=object)
    rng.shuffle(shuffled)

    half = len(shuffled) // 2
    hi_groups = set(shuffled[:half].tolist())
    out["assigned_target_lang"] = out[group_col].apply(
        lambda g: "hi" if g in hi_groups else "bn"
    )
    return out


def run_translation(
    input_csv: Path,
    target: str = "both",
    limit: int | None = None,
    batch_size: int = 4,
    backend: str = "hf",
    seed: int = 42,
    translator_factory: Any = None,
    output_csv: Path | None = None,
    cache_path: Path = CACHE_PATH,
) -> pd.DataFrame:
    """Translate assigned samples into Hindi and/or Bengali with caching, resume, and verification."""
    ensure_directories()
    logger = _setup_logger()

    if not input_csv.exists():
        raise FileNotFoundError(
            f"Input dataset not found at {input_csv}. Run `python -m ml.dataset.prepare --extract-source-only` first."
        )

    df = pd.read_csv(input_csv, keep_default_na=False)
    if "assigned_target_lang" not in df.columns:
        df = assign_translation_targets(df, seed=seed)

    targets = ["hi", "bn"] if target == "both" else [target]
    for t in targets:
        if t not in ("hi", "bn"):
            raise ValueError(f"Unsupported translation target '{t}'. Use 'hi', 'bn', or 'both'.")

    out_path = output_csv if output_csv is not None else (TRANSLATED_DIR / "climate_stance_translated.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing_map: dict[str, dict[str, Any]] = {}
    if out_path.exists():
        try:
            prev_df = pd.read_csv(out_path, keep_default_na=False)
            for _, row in prev_df.iterrows():
                if row.get("translation_status") == "Translated":
                    existing_map[f"{row['original_id']}::{row['language_code']}"] = row.to_dict()
        except Exception:
            pass

    cache = load_cache(cache_path)
    to_translate = df[df["assigned_target_lang"].isin(targets)].copy()
    if limit is not None and limit > 0:
        if target == "both":
            half_lim = max(1, limit // 2)
            hi_slice = to_translate[to_translate["assigned_target_lang"] == "hi"].head(half_lim)
            bn_slice = to_translate[to_translate["assigned_target_lang"] == "bn"].head(limit - len(hi_slice))
            to_translate = pd.concat([hi_slice, bn_slice], ignore_index=True)
        else:
            to_translate = to_translate.head(limit)

    selected_orig_ids = set(to_translate["id"].astype(str).tolist())
    english_df = df[df["id"].astype(str).isin(selected_orig_ids)] if (limit is not None and limit > 0) else df

    results: list[dict[str, Any]] = []
    for _, row in english_df.iterrows():
        results.append(
            {
                "id": f"{row['id']}_en",
                "original_id": str(row["id"]),
                "group_id": str(row.get("group_id", row["id"])),
                "original_text": str(row["original_text"]),
                "text": str(row["original_text"]),
                "language": "English",
                "language_code": "en",
                "label": str(row["label"]),
                "original_label": str(row.get("original_label", row["label"])),
                "original_source": str(row.get("original_source", "unknown")),
                "translation_language": "en",
                "translation_status": "Original",
                "is_machine_translated": False,
                "split": str(row.get("split", "")),
            }
        )

    pending_by_lang: dict[str, list[dict[str, Any]]] = {"hi": [], "bn": []}
    cached_count = 0
    translated_count = 0
    failed_count = 0

    for _, row in to_translate.iterrows():
        row_dict = row.to_dict()
        t_code = str(row_dict["assigned_target_lang"])
        lang_name = "Hindi" if t_code == "hi" else "Bengali"
        orig_id = str(row_dict["id"])
        orig_text = str(row_dict["original_text"])

        resume_key = f"{orig_id}::{t_code}"
        if resume_key in existing_map:
            results.append(existing_map[resume_key])
            cached_count += 1
            continue

        c_key = _cache_key(orig_text, t_code)
        if c_key in cache:
            results.append(
                {
                    "id": f"{orig_id}_{t_code}",
                    "original_id": orig_id,
                    "group_id": str(row_dict.get("group_id", orig_id)),
                    "original_text": orig_text,
                    "text": cache[c_key],
                    "language": lang_name,
                    "language_code": t_code,
                    "label": str(row_dict["label"]),
                    "original_label": str(row_dict.get("original_label", row_dict["label"])),
                    "original_source": str(row_dict.get("original_source", "unknown")),
                    "translation_language": t_code,
                    "translation_status": "Translated",
                    "is_machine_translated": True,
                    "split": str(row_dict.get("split", "")),
                }
            )
            cached_count += 1
        else:
            pending_by_lang[t_code].append(row_dict)

    print(
        f"Translation summary before run: cached={cached_count}, pending_hi={len(pending_by_lang['hi'])}, pending_bn={len(pending_by_lang['bn'])}"
    )

    for t_code in targets:
        pending_rows = pending_by_lang[t_code]
        if not pending_rows:
            continue
        lang_name = "Hindi" if t_code == "hi" else "Bengali"

        if backend == "hf" and translator_factory is None:
            texts = [str(r["original_text"]) for r in pending_rows]

            def _flush_batch(start_idx: int, batch_out: list[tuple[str | None, str | None]]) -> None:
                updated = False
                for offset, (tr_text, _) in enumerate(batch_out):
                    if tr_text is not None:
                        orig_t = str(pending_rows[start_idx + offset]["original_text"])
                        cache[_cache_key(orig_t, t_code)] = tr_text
                        updated = True
                if updated and ((start_idx // max(1, batch_size)) % 2 == 0 or start_idx + len(batch_out) >= len(pending_rows)):
                    save_cache(cache, cache_path)

            hf_outs = translate_batch_hf(
                texts,
                t_code,
                batch_size=batch_size,
                num_beams=1,
                on_batch_done=_flush_batch,
            )
            for row_dict, (tr_text, err) in zip(pending_rows, hf_outs):
                orig_id = str(row_dict["id"])
                orig_text = str(row_dict["original_text"])
                if tr_text is not None:
                    cache[_cache_key(orig_text, t_code)] = tr_text
                    translated_count += 1
                    results.append(
                        {
                            "id": f"{orig_id}_{t_code}",
                            "original_id": orig_id,
                            "group_id": str(row_dict.get("group_id", orig_id)),
                            "original_text": orig_text,
                            "text": tr_text,
                            "language": lang_name,
                            "language_code": t_code,
                            "label": str(row_dict["label"]),
                            "original_label": str(row_dict.get("original_label", row_dict["label"])),
                            "original_source": str(row_dict.get("original_source", "unknown")),
                            "translation_language": t_code,
                            "translation_status": "Translated",
                            "is_machine_translated": True,
                            "split": str(row_dict.get("split", "")),
                        }
                    )
                else:
                    failed_count += 1
                    logger.error("Translation failed for id=%s target=%s: %s", orig_id, t_code, err)
            save_cache(cache, cache_path)
        else:
            custom_tr = translator_factory(t_code) if translator_factory else None
            for row_dict in pending_rows:
                orig_id = str(row_dict["id"])
                orig_text = str(row_dict["original_text"])
                tr_text, err = translate_with_retry(
                    orig_text, t_code, translator=custom_tr, backend=backend
                )
                if tr_text is not None:
                    cache[_cache_key(orig_text, t_code)] = tr_text
                    translated_count += 1
                    results.append(
                        {
                            "id": f"{orig_id}_{t_code}",
                            "original_id": orig_id,
                            "group_id": str(row_dict.get("group_id", orig_id)),
                            "original_text": orig_text,
                            "text": tr_text,
                            "language": lang_name,
                            "language_code": t_code,
                            "label": str(row_dict["label"]),
                            "original_label": str(row_dict.get("original_label", row_dict["label"])),
                            "original_source": str(row_dict.get("original_source", "unknown")),
                            "translation_language": t_code,
                            "translation_status": "Translated",
                            "is_machine_translated": True,
                            "split": str(row_dict.get("split", "")),
                        }
                    )
                else:
                    failed_count += 1
                    logger.error("Translation failed for id=%s target=%s: %s", orig_id, t_code, err)
            save_cache(cache, cache_path)

    out_df = pd.DataFrame(results)
    if len(out_df) > 0:
        out_df = out_df.sort_values("id").reset_index(drop=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8")
    print(
        f"Saved translated dataset ({len(out_df):,} rows | new={translated_count}, cached={cached_count}, failed={failed_count}) -> {out_path}"
    )
    return out_df


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Translate half into Hindi and half into Bengali.")
    parser.add_argument("--target", choices=["hi", "bn", "both"], default="both")
    parser.add_argument("--input", type=Path, default=PROCESSED_DIR / "climate_stance_source.csv")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--backend", choices=["hf", "mymemory", "google", "auto"], default="hf")
    args = parser.parse_args()
    run_translation(
        input_csv=args.input,
        target=args.target,
        limit=args.limit,
        batch_size=args.batch_size,
        backend=args.backend,
        seed=int(cfg.get("random_seed", 42)),
    )


if __name__ == "__main__":
    main()
