"""Curated multilingual climate stance seed dataset (fallback only if raw Pol_NLI is unavailable)."""
from __future__ import annotations

import pandas as pd


_SEED_ROWS = [
    ("seed_01", "Climate change is an urgent existential crisis requiring immediate cuts in fossil fuel emissions.", "Favour", "English", "en"),
    ("seed_02", "Rising global temperatures are melting glaciers and threatening coastal cities worldwide.", "Favour", "English", "en"),
    ("seed_03", "Transitioning to solar and wind power is essential to stop dangerous global warming.", "Favour", "English", "en"),
    ("seed_04", "Global warming is exaggerated by media outlets and climate policies hurt economic growth.", "Against", "English", "en"),
    ("seed_05", "Climate has always changed naturally over centuries; human emissions are not a crisis.", "Against", "English", "en"),
    ("seed_06", "Carbon taxes punish working families without meaningfully changing global temperatures.", "Against", "English", "en"),
    ("seed_07", "The meteorological department forecast light rain in parts of Delhi tomorrow evening.", "None", "English", "en"),
    ("seed_08", "Environmental ministers met on Tuesday to review annual municipal waste reports.", "None", "English", "en"),
    ("seed_09", "University students organized a weekend seminar on urban architecture.", "None", "English", "en"),
    ("seed_01", "जलवायु परिवर्तन एक गंभीर संकट है और हमें जीवाश्म ईंधन उत्सर्जन तुरंत कम करना चाहिए।", "Favour", "Hindi", "hi"),
    ("seed_02", "बढ़ते वैश्विक तापमान से ग्लेशियर पिघल रहे हैं और तटीय शहरों को खतरा बढ़ रहा है।", "Favour", "Hindi", "hi"),
    ("seed_03", "ग्लोबल वार्मिंग को रोकने के लिए सौर और पवन ऊर्जा को अपनाना बेहद ज़रूरी है।", "Favour", "Hindi", "hi"),
    ("seed_04", "ग्लोबल वार्मिंग को मीडिया बढ़ा-चढ़ाकर दिखाता है और जलवायु नीतियां अर्थव्यवस्था को नुकसान पहुंचाती हैं।", "Against", "Hindi", "hi"),
    ("seed_05", "जलवायु सदियों से प्राकृतिक रूप से बदलती रही है, मानव उत्सर्जन कोई संकट नहीं है।", "Against", "Hindi", "hi"),
    ("seed_06", "कार्बन टैक्स आम परिवारों पर बोझ डालता है और तापमान पर कोई असर नहीं पड़ता।", "Against", "Hindi", "hi"),
    ("seed_07", "मौसम विभाग ने कल शाम दिल्ली के कुछ हिस्सों में हल्की बारिश का अनुमान जताया है।", "None", "Hindi", "hi"),
    ("seed_08", "मंगलवार को अधिकारियों ने वार्षिक नगर निगम कचरा रिपोर्ट की समीक्षा की।", "None", "Hindi", "hi"),
    ("seed_09", "विश्वविद्यालय के छात्रों ने शहरी वास्तुकला पर सप्ताहांत संगोष्ठी का आयोजन किया।", "None", "Hindi", "hi"),
    ("seed_01", "জলবায়ু পরিবর্তন একটি গুরুতর সংকট এবং জীবাশ্ম জ্বালানির নির্গমন অবিলম্বে কমানো উচিত।", "Favour", "Bengali", "bn"),
    ("seed_02", "বৈশ্বিক উষ্ণতা বৃদ্ধির ফলে হিমবাহ গলছে এবং উপকূলীয় শহরগুলো ঝুঁকির মুখে পড়ছে।", "Favour", "Bengali", "bn"),
    ("seed_03", "গ্লোবাল ওয়ার্মিং রোধ করতে সৌর ও বায়ু শক্তির ব্যবহার বাড়ানো অত্যন্ত জরুরি।", "Favour", "Bengali", "bn"),
    ("seed_04", "মিডিয়া গ্লোবাল ওয়ার্মিংকে অতিরঞ্জিত করে দেখায় এবং জলবায়ু নীতি অর্থনীতির ক্ষতি করে।", "Against", "Bengali", "bn"),
    ("seed_05", "জলবায়ু শতাব্দীর পর শতাব্দী ধরে প্রাকৃতিকভাবেই পরিবর্তিত হয়েছে, এটি কোনো মানবসৃষ্ট সংকট নয়।", "Against", "Bengali", "bn"),
    ("seed_06", "কার্বন কর সাধারণ পরিবারের ওপর চাপ সৃষ্টি করে কিন্তু বৈশ্বিক তাপমাত্রা বদলায় না।", "Against", "Bengali", "bn"),
    ("seed_07", "আবহাওয়া দপ্তর আগামীকাল সন্ধ্যায় কলকাতার কিছু অংশে হালকা বৃষ্টির পূর্বাভাস দিয়েছে।", "None", "Bengali", "bn"),
    ("seed_08", "মঙ্গলবার কর্মকর্তারা বার্ষিক পৌর বর্জ্য ব্যবস্থাপনা প্রতিবেদন পর্যালোচনা করেছেন।", "None", "Bengali", "bn"),
    ("seed_09", "বিশ্ববিদ্যালয়ের শিক্ষার্থীরা নগর স্থাপত্য নিয়ে একটি সাপ্তাহিক সেমিনার আয়োজন করেছে।", "None", "Bengali", "bn"),
]


def build_seed_source_dataframe() -> pd.DataFrame:
    en_rows = [r for r in _SEED_ROWS if r[4] == "en"]
    return pd.DataFrame(
        [
            {
                "id": gid,
                "group_id": gid,
                "original_text": txt,
                "text": txt,
                "language": lang,
                "language_code": code,
                "label": lbl,
                "original_label": lbl,
                "original_source": "curated_multilingual_seed",
                "translation_status": "Original",
                "is_machine_translated": False,
            }
            for gid, txt, lbl, lang, code in en_rows
        ]
    )


def build_seed_multilingual_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": f"{gid}_{code}",
                "original_id": gid,
                "group_id": gid,
                "original_text": txt,
                "text": txt,
                "language": lang,
                "language_code": code,
                "label": lbl,
                "original_label": lbl,
                "original_source": "curated_multilingual_seed",
                "translation_language": code,
                "translation_status": "Original" if code == "en" else "Translated",
                "is_machine_translated": False,
            }
            for gid, txt, lbl, lang, code in _SEED_ROWS
        ]
    )
