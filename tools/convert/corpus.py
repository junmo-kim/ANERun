"""Public evaluation corpus (multilingual, long). Writes <out_dir>/corpus.json [{name, lang, license, source, text}].
Usage: corpus.py <sources_dir> <out_dir>   (sources_dir: the files from `fetch.py data`, pinned in sources.json)
The document order is fixed: verify.py, verify_hf.py and scripts/accuracy/reference.py pick documents by index."""
import hashlib
import json
from pathlib import Path
import os
import re
import sys

SRC = [  # (name, lang, license, file in sources_dir, source label)
    ("omlx-readme-en", "en", "Apache-2.0", "omlx-README.md", "github.com/jundot/omlx README.md"),
    ("omlx-readme-ko", "ko", "Apache-2.0", "omlx-README.ko.md", "github.com/jundot/omlx README.ko.md"),
    ("omlx-readme-ja", "ja", "Apache-2.0", "omlx-README.ja.md", "github.com/jundot/omlx README.ja.md"),
    ("omlx-readme-zh", "zh", "Apache-2.0", "omlx-README.zh.md", "github.com/jundot/omlx README.zh.md"),
    ("omlx-readme-fr", "fr", "Apache-2.0", "omlx-README.fr.md", "github.com/jundot/omlx README.fr.md"),
    ("ko-constitution", "ko", "public domain", "ko_constitution.wiki", "ko.wikisource.org 대한민국헌법 (제10호)"),
    ("ja-constitution", "ja", "public domain", "ja_constitution.wiki", "ja.wikisource.org 日本國憲法"),
    ("sdcpp-server-api", "en", "MIT", "sdcpp-server-api.md", "github.com/leejet/stable-diffusion.cpp"),
    ("ggml-gguf-spec", "en", "MIT", "ggml-gguf.md", "github.com/ggml-org/ggml docs/gguf.md"),
    ("llamacpp-server-readme", "en", "MIT", "llamacpp-server-README.md", "github.com/ggml-org/llama.cpp"),
    ("llamacpp-build", "en", "MIT", "llamacpp-build.md", "github.com/ggml-org/llama.cpp"),
]


def clean_wiki(t):
    t = re.sub(r"\{\{[^{}]*\}\}", "", t)
    t = re.sub(r"\{\{[^{}]*\}\}", "", t)
    t = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", t)
    t = re.sub(r"<[^>]+>", "", t)
    t = re.sub(r"'''?", "", t)
    return re.sub(r"\n{3,}", "\n\n", t).strip()


def main(src_dir, out):
    docs = []
    for name, lang, lic, file, src in SRC:
        path = os.path.join(src_dir, file)
        with open(path, encoding="utf-8") as source:
            t = source.read()
        if path.endswith(".wiki"):
            t = clean_wiki(t)
        docs.append({"name": name, "lang": lang, "license": lic, "source": src, "text": t})
    here = Path(__file__).resolve()
    corpus = json.loads(here.with_name("sources.json").read_text())["corpus"]
    code = corpus["code_fixture"]
    parts = []
    for entry in code["sources"]:
        payload = (here.parents[2] / entry["path"]).read_bytes()   # frozen copies: the corpus does not follow code edits
        if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
            raise ValueError(f"{entry['path']} differs from the pinned {corpus['version']} fixture")
        parts.append(payload.decode("utf-8"))
    docs.append({"name": code["name"], "lang": "code", "license": code["license"],
                 "source": " + ".join(entry.get("origin", entry["path"]) for entry in code["sources"]), "text": "\n".join(parts)})
    for doc in docs:
        doc["corpus_version"] = corpus["version"]
    os.makedirs(out, exist_ok=True)
    json.dump(docs, open(f"{out}/corpus.json", "w", encoding="utf-8"), ensure_ascii=False)
    print(f"{len(docs)} docs -> {out}/corpus.json")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
