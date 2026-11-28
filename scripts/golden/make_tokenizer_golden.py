"""Golden token ids for the Swift tokenizer, from Hugging Face `tokenizers` (and cross-checked with
`transformers`).

Every string is generated here (seeded) or written for this file, so the fixture carries no third-party
text: authored sentences in many languages, random scalars from many Unicode blocks (CJK compatibility
ideographs, Hangul, kana, emoji with ZWJ / skin tones / flags / keycaps, combining marks, U+FEFF, control
characters, private use, astral planes), whitespace runs, special-token look-alikes, code (including the
Swift sources of this repository) and long mixed documents.

Usage (isolated venv, see CONTRIBUTING):
    uv venv /tmp/aneembed-golden && VIRTUAL_ENV=/tmp/aneembed-golden uv pip install tokenizers transformers
    /tmp/aneembed-golden/bin/python -I scripts/golden/make_tokenizer_golden.py <tokenizer.json> \
        Tests/Fixtures/tokenizer-golden.jsonl.xz

    # extra local corpus (not stored): any UTF-8 file, one text per line or a JSONL of strings
    ... make_tokenizer_golden.py <tokenizer.json> /tmp/extra.jsonl --lines corpus.txt
"""
import argparse
import json
import random
import sys
import unicodedata
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("tokenizer")
parser.add_argument("out")
parser.add_argument("--lines", help="tokenize this file's lines instead of the generated corpus")
parser.add_argument("--seed", type=int, default=20261114)
args = parser.parse_args()

from tokenizers import Tokenizer, __version__ as tokenizers_version  # noqa: E402
import transformers  # noqa: E402
from transformers import PreTrainedTokenizerFast  # noqa: E402

rng = random.Random(args.seed)
REPO = Path(__file__).resolve().parents[2]

AUTHORED = [
    "The quick brown fox jumps over the lazy dog.",
    "task: search result | query: how do I reset my router?",
    "title: none | text: The Neural Engine runs fixed-shape graphs.",
    "임베딩 모델은 문장을 고정 길이 벡터로 바꿉니다.",
    "오늘 서울의 날씨는 맑고, 오후에는 바람이 조금 붑니다.",
    "한국어와 English가 섞인 문장도 tokenizer가 처리해야 합니다.",
    "日本語の文章を正しく分割できるかを確認します。",
    "東京駅から新宿駅までは中央線で約十五分です。",
    "我们正在测试中文分词是否与参考实现完全一致。",
    "這是一個繁體中文的句子，包含標點符號。",
    "Привет, мир! Это проверка кириллицы.",
    "Γειά σου Κόσμε, αυτή είναι μια δοκιμή.",
    "مرحبا بالعالم، هذا اختبار للنص العربي.",
    "שלום עולם, זו בדיקה של טקסט בעברית.",
    "नमस्ते दुनिया, यह हिंदी पाठ का परीक्षण है।",
    "สวัสดีชาวโลก นี่คือการทดสอบภาษาไทย",
    "Xin chào thế giới, đây là bài kiểm tra tiếng Việt.",
    "Merhaba dünya, bu bir Türkçe denemesidir.",
    "Zażółć gęślą jaźń — polskie znaki diakrytyczne.",
    "Ça va très bien, merci ! L'été dernier à Paris.",
    "Größenwahn und Straße: ä ö ü ß ẞ.",
    "¿Dónde está la biblioteca? ¡Allí está!",
    "Olá, mundo! Ação, coração, pão.",
    "გამარჯობა მსოფლიო",
    "Բարեւ աշխարհ",
    "ሰላም ልዑል",
    "ជំរាបសួរពិភពលោក",
    "မင်္ဂလာပါကမ္ဘာလောက",
    "ಹಲೋ ವರ್ಲ್ಡ್ ತಮಿழ் வணக்கம் తెలుగు নমস্কার ਸਤ ਸ੍ਰੀ ਅਕਾਲ",
    "Sawubona mhlaba; Habari dunia; Kamusta mundo.",
    "𝔘𝔫𝔦𝔠𝔬𝔡𝔢 𝕞𝕒𝕥𝕙 𝓈𝒸𝓇𝒾𝓅𝓉 and ℌ𝔦𝔩𝔟𝔢𝔯𝔱",
    "E = mc² and ∑ᵢ xᵢ² ≤ ∞, ∀x ∈ ℝ: x² ≥ 0",
    "Price: ₩12,000 / $9.99 / €8,50 / ¥1,200 / £7",
    "https://example.com/path?query=a%20b&x=1#frag",
    "mailto:someone@example.org, user.name+tag@sub.example.co.kr",
    "| col1 | col2 |\n|------|------|\n| a    | b    |",
    "# Heading\n\n- item one\n- item two\n\n```swift\nlet x = 1\n```",
    "👨‍👩‍👧‍👦 family, 👩🏽‍💻 coder, 🏳️‍🌈 flag, 🇰🇷🇯🇵🇺🇸 flags, 1️⃣ keycap",
    "☺︎ vs ☺️, ❤ vs ❤️, ✌🏻✌🏼✌🏽✌🏾✌🏿",
    "﻿BOM at the start", "BOM in the ﻿ middle", "BOM at the end﻿", "﻿﻿﻿",
    "豈 更 車 賈 滑 串 句 龜 (compatibility) vs 豈更車賈滑串句龜",
    "한글 NFC vs " + unicodedata.normalize("NFD", "한글 NFD"),
    "é vs é, å vs å, ñ vs ñ",
    "Z͑ͫ̓ͪ̂ͫ̽͏̴̙a̷l̴g̵o",
    "<bos>hello<eos>", "<bos><bos><eos>", "<<bos>>", "<bo s>", "<|image>x<image|>", "<|image|>", "<|image|><|audio|>",
    "<start_of_image>", "<|tool>call<tool|>", "<|\"|>quoted<|\"|>", "<pad><mask><unk>", "<|turn>user\nhi<turn|>",
    "", " ", "  ", "\n", "\t", "\r\n", "a", "▁", "▁▁", " ▁ ", "▁hello", " hello", "hello ", "  hello  world  ",
]

BLOCKS = {
    "ascii": [(0x20, 0x7E)],
    "latin": [(0xA0, 0x24F), (0x1E00, 0x1EFF)],
    "combining": [(0x300, 0x36F), (0x1AB0, 0x1AFF), (0x20D0, 0x20FF)],
    "greek_cyrillic": [(0x370, 0x3FF), (0x400, 0x52F)],
    "hebrew_arabic": [(0x590, 0x5FF), (0x600, 0x6FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)],
    "indic": [(0x900, 0x97F), (0x980, 0x9FF), (0xB80, 0xBFF), (0xC00, 0xC7F), (0xD00, 0xD7F)],
    "thai_lao_tibetan": [(0xE00, 0xE7F), (0xE80, 0xEFF), (0xF00, 0xFFF)],
    "hangul": [(0xAC00, 0xD7A3), (0x1100, 0x11FF), (0x3130, 0x318F)],
    "kana": [(0x3040, 0x309F), (0x30A0, 0x30FF), (0xFF65, 0xFF9F)],
    "cjk": [(0x4E00, 0x9FFF), (0x3400, 0x4DBF)],
    "cjk_compat": [(0xF900, 0xFAFF), (0x2F800, 0x2FA1F), (0x3300, 0x33FF)],
    "cjk_ext": [(0x20000, 0x2A6DF), (0x2B740, 0x2B81F)],
    "fullwidth_punct": [(0xFF01, 0xFF5E), (0x3000, 0x303F), (0x2000, 0x206F)],
    "symbols": [(0x2100, 0x23FF), (0x2500, 0x27BF), (0x2900, 0x2BFF)],
    "emoji": [(0x1F300, 0x1F5FF), (0x1F600, 0x1F64F), (0x1F680, 0x1F6FF), (0x1F900, 0x1F9FF), (0x1FA70, 0x1FAFF)],
    "math_alnum": [(0x1D400, 0x1D7FF)],
    "control": [(0x00, 0x1F), (0x7F, 0x9F)],
    "private_use": [(0xE000, 0xF8FF), (0xF0000, 0xF00FF)],
    "specials": [(0xFFF0, 0xFFFD), (0xFE00, 0xFE0F), (0xE0100, 0xE01EF), (0xE0000, 0xE007F)],
    "astral_misc": [(0x10000, 0x1007F), (0x10300, 0x1034F), (0x13000, 0x1342F), (0x16800, 0x16A3F)],
}
ALL_RANGES = [r for ranges in BLOCKS.values() for r in ranges]


def scalar(ranges):
    lo, hi = rng.choice(ranges)
    while True:
        c = rng.randint(lo, hi)
        if not 0xD800 <= c <= 0xDFFF:
            return chr(c)


def random_string(ranges, n):
    return "".join(scalar(ranges) for _ in range(n))


SKIN = ["", "\U0001F3FB", "\U0001F3FC", "\U0001F3FD", "\U0001F3FE", "\U0001F3FF"]
PEOPLE = ["\U0001F468", "\U0001F469", "\U0001F467", "\U0001F466", "\U0001F9D1"]
OBJECTS = ["\U0001F4BB", "\U0001F680", "\U0001F373", "\U0001F527", "⚕️", "\U0001F3A8", "\U0001F9AF"]


def emoji_sequence():
    kind = rng.randrange(5)
    if kind == 0:
        return "‍".join(rng.choice(PEOPLE) for _ in range(rng.randint(2, 4)))
    if kind == 1:
        return rng.choice(PEOPLE) + rng.choice(SKIN) + "‍" + rng.choice(OBJECTS)
    if kind == 2:
        return "".join(chr(0x1F1E6 + rng.randrange(26)) for _ in range(2 * rng.randint(1, 3)))
    if kind == 3:
        return rng.choice("0123456789#*") + "️⃣"
    return "\U0001F3F4" + "".join(chr(0xE0000 + ord(c)) for c in rng.choice(["gbeng", "gbsct", "gbwls"])) + "\U000E007F"


WHITESPACE = [" ", "\n", "\t", "\r", "\r\n", "　", " ", " ", "​", " ", "\x0b", "\x0c"]


def whitespace_run():
    if rng.random() < 0.5:
        return rng.choice(WHITESPACE) * rng.randint(1, 80)
    return "".join(rng.choice(WHITESPACE) for _ in range(rng.randint(1, 40)))


def combining():
    base = scalar(BLOCKS["latin"] + BLOCKS["ascii"] + BLOCKS["hangul"])
    return base + "".join(scalar(BLOCKS["combining"]) for _ in range(rng.randint(1, 6)))


SPECIALS = ["<pad>", "<eos>", "<bos>", "<unk>", "<mask>", "<|tool>", "<tool|>", "<|image>", "<image|>",
            "<|image|>", "<|audio>", "<|audio|>", "<|video|>", "<|turn>", "<turn|>", "<|\"|>", "<|think|>"]


def special_mix():
    parts = []
    for _ in range(rng.randint(1, 6)):
        r = rng.random()
        if r < 0.4:
            parts.append(rng.choice(SPECIALS))
        elif r < 0.6:
            s = rng.choice(SPECIALS)
            parts.append(s[: rng.randint(1, len(s) - 1)])  # partial look-alike
        else:
            parts.append(random_string(BLOCKS["ascii"] + BLOCKS["hangul"], rng.randint(0, 6)))
    return "".join(parts)


def code_lines():
    lines = []
    for path in sorted((REPO / "Sources").rglob("*.swift")):
        lines += [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    lines += [
        "def f(x):\n    return {k: v for k, v in x.items() if v is not None}",
        "const add = (a, b) => a + b; // arrow function",
        "SELECT id, name FROM users WHERE created_at > NOW() - INTERVAL '7 days';",
        "<div class=\"card\"><p>Hello &amp; welcome</p></div>",
        '{"input": ["a", "b"], "encoding_format": "base64", "dimensions": 256}',
        "fn main() { println!(\"{}\", 1 + 2); }",
        "#include <stdio.h>\nint main(void) { printf(\"%d\\n\", 42); return 0; }",
        "    \t\tindented with tabs and spaces\t \t",
    ]
    return lines


def corpus():
    texts = list(AUTHORED)
    for name, ranges in BLOCKS.items():
        for _ in range(300):
            texts.append(random_string(ranges, rng.randint(1, 32)))
    texts += ["".join(chr(c) for c in range(start, min(start + 32, 0xFADA))) for start in range(0xF900, 0xFADA, 32)]
    texts += ["".join(chr(c) for c in range(start, start + 32)) for start in range(0x2F800, 0x2FA1E, 32)]
    texts += [emoji_sequence() for _ in range(600)]
    texts += [" ".join(emoji_sequence() for _ in range(rng.randint(2, 6))) for _ in range(150)]
    texts += [whitespace_run() for _ in range(500)]
    texts += [combining() * rng.randint(1, 4) for _ in range(300)]
    texts += [special_mix() for _ in range(600)]
    code = code_lines()
    texts += code
    texts += ["\n".join(rng.sample(code, rng.randint(3, 12))) for _ in range(60)]
    # mixed documents: authored sentences, random blocks, whitespace, emoji and BOMs
    def piece():
        r = rng.random()
        if r < 0.35:
            return rng.choice(AUTHORED)
        if r < 0.6:
            return random_string(rng.choice(list(BLOCKS.values())), rng.randint(1, 30))
        if r < 0.75:
            return whitespace_run()
        if r < 0.85:
            return emoji_sequence()
        if r < 0.92:
            return "﻿"
        return rng.choice(code)
    texts += ["".join(piece() for _ in range(rng.randint(2, 20))) for _ in range(1500)]
    texts += [random_string(ALL_RANGES, rng.randint(1, 60)) for _ in range(2000)]
    # long single words (no spaces survive normalization): BPE over whole documents
    for n in (2_000, 8_000, 20_000):
        texts.append(" ".join(rng.choice(AUTHORED) for _ in range(n // 40)))
        texts.append("".join(piece() for _ in range(n // 20)))
    texts.append("a" * 5000)
    texts.append(" " * 3000)
    texts.append("ab" * 4000)
    texts.append("가" * 3000)
    return texts


if args.lines:
    raw = Path(args.lines).read_text(encoding="utf-8")
    texts = [json.loads(l) if l.startswith('"') else l for l in raw.splitlines() if l]
else:
    texts = corpus()

tokenizer = Tokenizer.from_file(args.tokenizer)
fast = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer)
out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
seen = set()
written = disagreements = tokens = 0
import io, lzma
buffer = io.StringIO()
with buffer as handle:
    for text in texts:
        if text in seen:
            continue
        seen.add(text)
        ids = tokenizer.encode(text).ids
        if fast(text)["input_ids"] != ids:
            disagreements += 1
        handle.write(json.dumps({"text": text, "ids": ids}, ensure_ascii=True) + "\n")
        written += 1
        tokens += len(ids)
    data = handle.getvalue().encode("utf-8")
out.write_bytes(lzma.compress(data, format=lzma.FORMAT_XZ) if out.suffix == ".xz" else data)
print(f"{written} texts, {tokens} ids, tokenizers {tokenizers_version}, transformers {transformers.__version__}, "
      f"transformers disagreements {disagreements}", file=sys.stderr)
