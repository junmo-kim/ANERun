"""Rebuild the fixed Qwen HF corpus and model fixture from pinned public sources."""
import argparse
import hashlib
import json
import lzma
import random
import subprocess
import unicodedata
from pathlib import Path

import tokenizers
from tokenizers import Regex, Tokenizer, pre_tokenizers

ROOT = Path(__file__).resolve().parents[3]
REVISION = "b1044567ca6a59ba467c56f9db50dfe91bed2656"
MODEL_SHA256 = "c6fb5c5bbba5fa5f8332edfb6d8aa67bd7fb3d75365b1765f108201698eaebf5"
LONG_SHA256 = "4e474127d0240addf712819bcb90bc583e95037f970aa05918ec55873a2e25b3"
GOLDEN_SHA256 = "8830d983af6e957dd7a4a2b2cefaa876fd3f646dc5969cfb7875aee9a8953731"
SEED = 20261118


def verify(data, expected):
    digest = hashlib.sha256(data).hexdigest()
    if digest != expected:
        raise SystemExit(f"SHA-256 mismatch: expected {expected}, got {digest}")


def long_example():
    # The spike's 32,760-ID row decodes to this byte slice of public ANEEmbed v0.1.1.
    def git(*args):
        return subprocess.check_output(["git", "-C", str(ROOT), *args])

    files = git("ls-tree", "-r", "--name-only", REVISION, "Sources/EmbedCore").decode().splitlines()
    first = files.index("Sources/EmbedCore/Image/JPEGDecoder.swift")
    last = files.index("Sources/EmbedCore/Service/EmbeddingService.swift")
    sources = b"".join(git("show", f"{REVISION}:{path}") for path in files[first:last + 1])
    data = sources[42515:42515 + 138301]
    verify(data, LONG_SHA256)
    return data.decode("utf-8")


def examples(config, long_text):
    fixed = {
        "empty": "", "korean": "한글 띄어쓰기와 받침입니다.",
        "ja-zh": "「東京都」\t中文简体與繁體。漢字かな交じり文",
        "contractions": "IT'S I'll WE'LL he's you'd've I'm isn't 'rE 'ſ",
        "emoji": "🇰🇷 🧑🏽‍💻👩‍👩‍👧‍👦1️⃣ ❤️‍🔥",
        "nfc": "e\u0301 é A\u030A Å Å 가 각 x\u0327\u0301 \u0344",
        "digits": "0123456789 １２３４ ١٢٣ Ⅲ ³፩",
        "swift-code": 'func f(_ s: String) -> String {\n  "👋 " + s\n}',
        "python-code": "def f(xs):\r\n\treturn [x**2 for x in xs if x % 3]\n",
        "json-code": '{"a":[true,null,32],"b":"안녕","c":"\\n"}',
        "spaces": " \t\r\n\r\n  \n  a\t  ! \u0085\u00a0\u2003\u2028\u2029\u3000  ",
        "controls": "\0\u000b\u000c\u001c\u001d\u001e\u001f\u007f\ufeff\uffff",
        "punctuation": "!@#$%\r\n\r\n👋\nnext\r\n",
        "unicode-class-contraction": "\U00032b40's",
        "nfc-astral-mark": "\U00101f10\u0327\u0301 \U000f00d8\u0301 \U000f1f30\u0301",
        "nfc-age": "\U000105d2\u0307 \U00011382\U000113c9 a\u0897\u0323",
        "contraction-fold": "'ſ 'S 's 'LL 'lL 'RE 'vE 'D 't 'M",
        "partial-special": "<|endoftext|<think <|ENDOFTeXT|> <||> </think",
        "overlaps": "a" * 963 + "!" * 1025 + " " * 8193,
    }
    for name, text in fixed.items():
        yield name, text
    literals = [x["content"] for x in config["added_tokens"]]
    for i, lit in enumerate(literals):
        yield f"literal-{i}", lit
        yield f"embedded-{i}", "a" + lit + lit + "\u0301한"
    yield "adjacent-literals", "".join(literals)
    yield "decomposed-literal", unicodedata.normalize("NFD", literals[0])
    yield "long32760", long_text
    rng = random.Random(SEED)
    mixed = "Aa' \r\n\t\0Ｚ١！,🙌🇰🇷‍❤\ufe0f汉あ한한" + "".join(
        chr(n) for n in (0x0301, 0x0327, 0x85, 0xA0, 0x2028, 0x200B, 0x1C, 0xFEFF)
    )

    def scalar():
        n = rng.randrange(0x110000 - 0x800)
        return chr(n if n < 0xD800 else n + 0x800)

    for i in range(4096):
        yield f"uniform-{i}", "".join(scalar() for _ in range(rng.randint(1, 48)))
    for i in range(2048):
        yield f"mixed-{i}", "".join(
            rng.choice(mixed) if rng.randrange(4) else scalar()
            for _ in range(rng.randint(1, 72))
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tokenizer_json", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    if tokenizers.__version__ != "0.23.3":
        raise SystemExit("use tokenizers==0.23.3")
    model = args.tokenizer_json.read_bytes()
    verify(model, MODEL_SHA256)
    config = json.loads(model)
    tokenizer = Tokenizer.from_str(model.decode("utf-8"))
    tokenizer.encode_special_tokens = False
    pattern = config["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
    splitter = pre_tokenizers.Split(Regex(pattern), behavior="isolated")

    rows = []
    for name, text in examples(config, long_example()):
        ids = tokenizer.encode(text, add_special_tokens=True).ids
        pieces = [piece for piece, _ in splitter.pre_tokenize_str(text)]
        assert "".join(pieces) == text
        rows.append(json.dumps(dict(name=name, text=text, ids=ids, pieces=pieces), separators=(",", ":")))
    golden = ("\n".join(rows) + "\n").encode("ascii")
    verify(golden, GOLDEN_SHA256)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in [("tokenizer.json.xz", model), ("golden.jsonl.xz", golden)]:
        (args.output_dir / name).write_bytes(lzma.compress(data, preset=6))


if __name__ == "__main__":
    main()
