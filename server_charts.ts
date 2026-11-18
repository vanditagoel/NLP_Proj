import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';

const FONT_5X7: Record<string, string[]> = {
  A: ['01110', '10001', '10001', '11111', '10001', '10001', '10001'],
  B: ['11110', '10001', '10001', '11110', '10001', '10001', '11110'],
  C: ['01110', '10001', '10000', '10000', '10000', '10001', '01110'],
  D: ['11100', '10010', '10001', '10001', '10001', '10010', '11100'],
  E: ['11111', '10000', '10000', '11110', '10000', '10000', '11111'],
  F: ['11111', '10000', '10000', '11110', '10000', '10000', '10000'],
  G: ['01110', '10001', '10000', '10111', '10001', '10001', '01110'],
  H: ['10001', '10001', '10001', '11111', '10001', '10001', '10001'],
  I: ['01110', '00100', '00100', '00100', '00100', '00100', '01110'],
  J: ['00111', '00010', '00010', '00010', '10010', '10010', '01100'],
  K: ['10001', '10010', '10100', '11000', '10100', '10010', '10001'],
  L: ['10000', '10000', '10000', '10000', '10000', '10000', '11111'],
  M: ['10001', '11011', '10101', '10101', '10001', '10001', '10001'],
  N: ['10001', '11001', '10101', '10011', '10001', '10001', '10001'],
  O: ['01110', '10001', '10001', '10001', '10001', '10001', '01110'],
  P: ['11110', '10001', '10001', '11110', '10000', '10000', '10000'],
  Q: ['01110', '10001', '10001', '10001', '10101', '10010', '01101'],
  R: ['11110', '10001', '10001', '11110', '10100', '10010', '10001'],
  S: ['01111', '10000', '10000', '01110', '00001', '00001', '11110'],
  T: ['11111', '00100', '00100', '00100', '00100', '00100', '00100'],
  U: ['10001', '10001', '10001', '10001', '10001', '10001', '01110'],
  V: ['10001', '10001', '10001', '10001', '10001', '01010', '00100'],
  W: ['10001', '10001', '10001', '10101', '10101', '11011', '10001'],
  X: ['10001', '10001', '01010', '00100', '01010', '10001', '10001'],
  Y: ['10001', '10001', '01010', '00100', '00100', '00100', '00100'],
  Z: ['11111', '00001', '00010', '00100', '01000', '10000', '11111'],
  a: ['00000', '00000', '01110', '00001', '01111', '10001', '01111'],
  b: ['10000', '10000', '11110', '10001', '10001', '10001', '11110'],
  c: ['00000', '00000', '01110', '10000', '10000', '10001', '01110'],
  d: ['00001', '00001', '01111', '10001', '10001', '10001', '01111'],
  e: ['00000', '00000', '01110', '10001', '11111', '10000', '01110'],
  f: ['00110', '01001', '01000', '11100', '01000', '01000', '01000'],
  g: ['00000', '01111', '10001', '10001', '01111', '00001', '01110'],
  h: ['10000', '10000', '11110', '10001', '10001', '10001', '10001'],
  i: ['00100', '00000', '01100', '00100', '00100', '00100', '01110'],
  j: ['00010', '00000', '00110', '00010', '00010', '10010', '01100'],
  k: ['10000', '10000', '10010', '10100', '11000', '10100', '10010'],
  l: ['01100', '00100', '00100', '00100', '00100', '00100', '01110'],
  m: ['00000', '00000', '11010', '10101', '10101', '10001', '10001'],
  n: ['00000', '00000', '11110', '10001', '10001', '10001', '10001'],
  o: ['00000', '00000', '01110', '10001', '10001', '10001', '01110'],
  p: ['00000', '00000', '11110', '10001', '11110', '10000', '10000'],
  q: ['00000', '00000', '01111', '10001', '01111', '00001', '00001'],
  r: ['00000', '00000', '10110', '11001', '10000', '10000', '10000'],
  s: ['00000', '00000', '01111', '10000', '01110', '00001', '11110'],
  t: ['01000', '01000', '11100', '01000', '01000', '01001', '00110'],
  u: ['00000', '00000', '10001', '10001', '10001', '10011', '01101'],
  v: ['00000', '00000', '10001', '10001', '10001', '01010', '00100'],
  w: ['00000', '00000', '10001', '10001', '10101', '10101', '01010'],
  x: ['00000', '00000', '10001', '01010', '00100', '01010', '10001'],
  y: ['00000', '00000', '10001', '10001', '01111', '00001', '01110'],
  z: ['00000', '00000', '11111', '00010', '00100', '01000', '11111'],
  '0': ['01110', '10001', '10011', '10101', '11001', '10001', '01110'],
  '1': ['00100', '01100', '00100', '00100', '00100', '00100', '01110'],
  '2': ['01110', '10001', '00001', '00110', '01000', '10000', '11111'],
  '3': ['11110', '00001', '00001', '01110', '00001', '00001', '11110'],
  '4': ['00010', '00110', '01010', '10010', '11111', '00010', '00010'],
  '5': ['11111', '10000', '11110', '00001', '00001', '10001', '01110'],
  '6': ['00110', '01000', '10000', '11110', '10001', '10001', '01110'],
  '7': ['11111', '00001', '00010', '00100', '01000', '01000', '01000'],
  '8': ['01110', '10001', '10001', '01110', '10001', '10001', '01110'],
  '9': ['01110', '10001', '10001', '01111', '00001', '00010', '01100'],
  '.': ['00000', '00000', '00000', '00000', '00000', '01100', '01100'],
  ',': ['00000', '00000', '00000', '00000', '00000', '00100', '01000'],
  ':': ['00000', '01100', '01100', '00000', '01100', '01100', '00000'],
  '-': ['00000', '00000', '00000', '11111', '00000', '00000', '00000'],
  '+': ['00000', '00100', '00100', '11111', '00100', '00100', '00000'],
  '/': ['00001', '00010', '00010', '00100', '01000', '01000', '10000'],
  '(': ['00010', '00100', '01000', '01000', '01000', '00100', '00010'],
  ')': ['01000', '00100', '00010', '00010', '00010', '00100', '01000'],
  '%': ['11001', '11010', '00010', '00100', '01000', '01011', '10011'],
  ' ': ['00000', '00000', '00000', '00000', '00000', '00000', '00000'],
};

type RGB = [number, number, number];

class PixelCanvas {
  readonly width: number;
  readonly height: number;
  readonly buf: Uint8Array;

  constructor(width: number, height: number, bg: RGB = [255, 255, 255]) {
    this.width = width;
    this.height = height;
    this.buf = new Uint8Array(width * height * 3);
    for (let i = 0; i < width * height; i++) {
      this.buf[i * 3] = bg[0];
      this.buf[i * 3 + 1] = bg[1];
      this.buf[i * 3 + 2] = bg[2];
    }
  }

  setPixel(x: number, y: number, color: RGB): void {
    const ix = Math.round(x);
    const iy = Math.round(y);
    if (ix < 0 || ix >= this.width || iy < 0 || iy >= this.height) return;
    const idx = (iy * this.width + ix) * 3;
    this.buf[idx] = color[0];
    this.buf[idx + 1] = color[1];
    this.buf[idx + 2] = color[2];
  }

  fillRect(x: number, y: number, w: number, h: number, color: RGB): void {
    const x0 = Math.max(0, Math.round(x));
    const y0 = Math.max(0, Math.round(y));
    const x1 = Math.min(this.width, Math.round(x + w));
    const y1 = Math.min(this.height, Math.round(y + h));
    for (let py = y0; py < y1; py++) {
      for (let px = x0; px < x1; px++) {
        const idx = (py * this.width + px) * 3;
        this.buf[idx] = color[0];
        this.buf[idx + 1] = color[1];
        this.buf[idx + 2] = color[2];
      }
    }
  }

  strokeRect(x: number, y: number, w: number, h: number, color: RGB): void {
    this.fillRect(x, y, w, 1, color);
    this.fillRect(x, y + h - 1, w, 1, color);
    this.fillRect(x, y, 1, h, color);
    this.fillRect(x + w - 1, y, 1, h, color);
  }

  drawText(
    text: string,
    x: number,
    y: number,
    color: RGB = [15, 23, 42],
    scale = 2,
    align: 'left' | 'center' | 'right' = 'left'
  ): void {
    const clean = text.replace(/[^\x20-\x7E]/g, ' ');
    const totalW = clean.length * 6 * scale - scale;
    let cursorX = x;
    if (align === 'center') cursorX = x - totalW / 2;
    if (align === 'right') cursorX = x - totalW;

    for (let cIdx = 0; cIdx < clean.length; cIdx++) {
      const ch = clean[cIdx];
      const glyph = FONT_5X7[ch] || FONT_5X7[ch.toUpperCase()] || FONT_5X7[' '];
      for (let row = 0; row < 7; row++) {
        for (let col = 0; col < 5; col++) {
          if (glyph[row][col] === '1') {
            this.fillRect(
              cursorX + col * scale,
              y + row * scale,
              scale,
              scale,
              color
            );
          }
        }
      }
      cursorX += 6 * scale;
    }
  }

  toPngBuffer(): Buffer {
    const raw = Buffer.alloc((this.width * 3 + 1) * this.height);
    for (let y = 0; y < this.height; y++) {
      const rowStart = y * (this.width * 3 + 1);
      raw[rowStart] = 0;
      const srcStart = y * this.width * 3;
      raw.set(this.buf.subarray(srcStart, srcStart + this.width * 3), rowStart + 1);
    }

    const compressed = zlib.deflateSync(raw, { level: 6 });

    const makeChunk = (type: string, data: Buffer): Buffer => {
      const len = Buffer.alloc(4);
      len.writeUInt32BE(data.length, 0);
      const typeBuf = Buffer.from(type, 'ascii');
      const crcInput = Buffer.concat([typeBuf, data]);
      const crcBuf = Buffer.alloc(4);
      crcBuf.writeUInt32BE(crc32(crcInput) >>> 0, 0);
      return Buffer.concat([len, typeBuf, data, crcBuf]);
    };

    const ihdr = Buffer.alloc(13);
    ihdr.writeUInt32BE(this.width, 0);
    ihdr.writeUInt32BE(this.height, 4);
    ihdr[8] = 8; // bit depth
    ihdr[9] = 2; // RGB
    ihdr[10] = 0;
    ihdr[11] = 0;
    ihdr[12] = 0;

    return Buffer.concat([
      Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]),
      makeChunk('IHDR', ihdr),
      makeChunk('IDAT', compressed),
      makeChunk('IEND', Buffer.alloc(0)),
    ]);
  }
}

const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) {
      c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    }
    table[n] = c;
  }
  return table;
})();

function crc32(buf: Buffer): number {
  let c = 0xffffffff;
  for (let i = 0; i < buf.length; i++) {
    c = CRC_TABLE[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
  }
  return (c ^ 0xffffffff) >>> 0;
}

const SLATE_900: RGB = [15, 23, 42];
const SLATE_600: RGB = [71, 85, 105];
const SLATE_200: RGB = [226, 232, 240];
const TEAL_700: RGB = [15, 118, 110];
const TEAL_600: RGB = [13, 148, 136];
const EMERALD_600: RGB = [5, 150, 105];
const ROSE_600: RGB = [225, 29, 72];
const SKY_600: RGB = [2, 132, 199];
const AMBER_600: RGB = [217, 119, 6];
const WHITE: RGB = [255, 255, 255];

export function saveBarChartPng(
  outPath: string,
  title: string,
  items: Array<{ label: string; value: number; color?: RGB }>
): void {
  const w = 800;
  const h = 440;
  const c = new PixelCanvas(w, h);
  c.drawText(title, w / 2, 22, SLATE_900, 2, 'center');

  const padL = 80;
  const padR = 40;
  const padT = 70;
  const padB = 75;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;
  const maxVal = Math.max(1, ...items.map((i) => i.value)) * 1.15;

  for (let t = 0; t <= 4; t++) {
    const val = Math.round((maxVal * t) / 4);
    const y = padT + plotH - (t / 4) * plotH;
    c.fillRect(padL, y, plotW, 1, SLATE_200);
    c.drawText(val.toLocaleString(), padL - 10, y - 6, SLATE_600, 2, 'right');
  }

  c.fillRect(padL, padT, 2, plotH, SLATE_600);
  c.fillRect(padL, padT + plotH, plotW, 2, SLATE_600);

  const slotW = plotW / Math.max(1, items.length);
  const barW = Math.min(110, slotW * 0.56);

  items.forEach((item, idx) => {
    const bx = padL + idx * slotW + (slotW - barW) / 2;
    const bh = (item.value / maxVal) * plotH;
    const by = padT + plotH - bh;
    c.fillRect(bx, by, barW, bh, item.color || TEAL_700);
    c.drawText(item.value.toLocaleString(), bx + barW / 2, by - 20, SLATE_900, 2, 'center');
    c.drawText(item.label, bx + barW / 2, padT + plotH + 16, SLATE_900, 2, 'center');
  });

  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  fs.writeFileSync(outPath, c.toPngBuffer());
}

export function saveGroupedBarChartPng(
  outPath: string,
  title: string,
  categories: string[],
  series: Array<{ name: string; values: number[]; color: RGB }>,
  ySuffix = ''
): void {
  const w = 860;
  const h = 470;
  const c = new PixelCanvas(w, h);
  c.drawText(title, w / 2, 20, SLATE_900, 2, 'center');

  // Legend
  let lx = 90;
  for (const s of series) {
    c.fillRect(lx, 50, 14, 14, s.color);
    c.drawText(s.name, lx + 22, 50, SLATE_900, 2, 'left');
    lx += 22 + s.name.length * 12 + 30;
  }

  const padL = 85;
  const padR = 35;
  const padT = 85;
  const padB = 80;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;
  const maxVal = ySuffix === '%' ? 105 : Math.max(1, ...series.flatMap((s) => s.values)) * 1.15;

  for (let t = 0; t <= 4; t++) {
    const val = Math.round((maxVal * t) / 4);
    const y = padT + plotH - (t / 4) * plotH;
    c.fillRect(padL, y, plotW, 1, SLATE_200);
    c.drawText(`${val}${ySuffix}`, padL - 10, y - 6, SLATE_600, 2, 'right');
  }

  c.fillRect(padL, padT, 2, plotH, SLATE_600);
  c.fillRect(padL, padT + plotH, plotW, 2, SLATE_600);

  const catW = plotW / Math.max(1, categories.length);
  const groupW = catW * 0.72;
  const barW = Math.min(46, groupW / Math.max(1, series.length) - 4);

  categories.forEach((cat, cIdx) => {
    const groupStart = padL + cIdx * catW + (catW - series.length * (barW + 4)) / 2;
    series.forEach((s, sIdx) => {
      const val = s.values[cIdx] ?? 0;
      const bx = groupStart + sIdx * (barW + 4);
      const bh = (val / maxVal) * plotH;
      const by = padT + plotH - bh;
      c.fillRect(bx, by, barW, bh, s.color);
      const lbl = ySuffix === '%' ? `${val.toFixed(0)}` : String(val);
      c.drawText(lbl, bx + barW / 2, Math.max(padT + 2, by - 16), SLATE_900, 1, 'center');
    });
    c.drawText(cat, padL + cIdx * catW + catW / 2, padT + plotH + 18, SLATE_900, 2, 'center');
  });

  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  fs.writeFileSync(outPath, c.toPngBuffer());
}

export function saveModelComparisonChartPng(
  outPath: string,
  models: Array<{ model_name: string; macro_f1: number; accuracy: number }>
): void {
  const w = 900;
  const h = Math.max(400, 130 + models.length * 76);
  const c = new PixelCanvas(w, h);
  c.drawText('Transformer Model Comparison (Held-Out Test Split)', w / 2, 20, SLATE_900, 2, 'center');

  c.fillRect(280, 50, 14, 14, TEAL_700);
  c.drawText('Macro F1 (%)', 302, 50, SLATE_900, 2, 'left');
  c.fillRect(480, 50, 14, 14, SKY_600);
  c.drawText('Accuracy (%)', 502, 50, SLATE_900, 2, 'left');

  const padL = 290;
  const padR = 65;
  const padT = 85;
  const padB = 55;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  for (let t = 0; t <= 5; t++) {
    const pct = t * 20;
    const x = padL + (pct / 100) * plotW;
    c.fillRect(x, padT, 1, plotH, SLATE_200);
    c.drawText(`${pct}%`, x, padT + plotH + 12, SLATE_600, 2, 'center');
  }
  c.fillRect(padL, padT, 2, plotH, SLATE_600);
  c.fillRect(padL, padT + plotH, plotW, 2, SLATE_600);

  const rowH = plotH / Math.max(1, models.length);
  models.forEach((m, idx) => {
    const cy = padT + idx * rowH + rowH / 2;
    const shortName = m.model_name.length > 22 ? m.model_name.slice(0, 22) : m.model_name;
    c.drawText(shortName, padL - 12, cy - 7, SLATE_900, 2, 'right');

    const f1W = Math.min(1, m.macro_f1) * plotW;
    const accW = Math.min(1, m.accuracy) * plotW;
    c.fillRect(padL + 2, cy - 18, f1W, 15, TEAL_700);
    c.drawText(`${(m.macro_f1 * 100).toFixed(1)}%`, padL + f1W + 8, cy - 16, SLATE_900, 2, 'left');

    c.fillRect(padL + 2, cy + 3, accW, 15, SKY_600);
    c.drawText(`${(m.accuracy * 100).toFixed(1)}%`, padL + accW + 8, cy + 5, SLATE_600, 2, 'left');
  });

  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  fs.writeFileSync(outPath, c.toPngBuffer());
}

export function saveConfusionMatrixChartPng(
  outPath: string,
  modelName: string,
  labels: string[],
  cm: number[][]
): void {
  const w = 620;
  const h = 500;
  const c = new PixelCanvas(w, h);
  const title = `CM: ${modelName.slice(0, 32)}`;
  c.drawText(title, w / 2, 22, SLATE_900, 2, 'center');
  c.drawText('True (rows) vs Predicted (cols)', w / 2, 48, SLATE_600, 2, 'center');

  const padL = 135;
  const padT = 100;
  const cell = 110;
  const maxVal = Math.max(1, ...cm.flat());

  labels.forEach((lbl, idx) => {
    c.drawText(lbl, padL + idx * cell + cell / 2, padT - 26, SLATE_900, 2, 'center');
    c.drawText(lbl, padL - 14, padT + idx * cell + cell / 2 - 7, SLATE_900, 2, 'right');
  });

  for (let r = 0; r < labels.length; r++) {
    for (let col = 0; col < labels.length; col++) {
      const val = cm[r]?.[col] ?? 0;
      const ratio = val / maxVal;
      const bg: RGB = [
        Math.round(240 - ratio * (240 - 15)),
        Math.round(253 - ratio * (253 - 118)),
        Math.round(250 - ratio * (250 - 110)),
      ];
      const cx = padL + col * cell;
      const cy = padT + r * cell;
      c.fillRect(cx, cy, cell, cell, bg);
      c.strokeRect(cx, cy, cell, cell, SLATE_200);
      const txtCol = ratio > 0.5 ? WHITE : SLATE_900;
      c.drawText(String(val), cx + cell / 2, cy + cell / 2 - 8, txtCol, 2, 'center');
    }
  }

  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  fs.writeFileSync(outPath, c.toPngBuffer());
}

export function saveHistogramPng(
  outPath: string,
  title: string,
  values: number[],
  bins = 12
): void {
  const minV = Math.min(...values, 0);
  const maxV = Math.max(...values, 1);
  const step = Math.max(1, Math.ceil((maxV - minV) / bins));
  const items: Array<{ label: string; value: number; color: RGB }> = [];
  for (let b = 0; b < bins; b++) {
    const lo = minV + b * step;
    const hi = lo + step;
    const cnt = values.filter((v) => v >= lo && (b === bins - 1 ? v <= hi : v < hi)).length;
    items.push({ label: `${lo}`, value: cnt, color: TEAL_600 });
  }
  saveBarChartPng(outPath, title, items);
}

export const CHART_COLORS = {
  TEAL_700,
  EMERALD_600,
  ROSE_600,
  SKY_600,
  AMBER_600,
  SLATE_600,
};
