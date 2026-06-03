# xmltokenizer

Robust inline tokenization of TEI/XML and generic XML. Designed primarily as a **library** for `flexipipe` (and any other Python tool that produces CoNLL-U), with first-class standalone modes for ad-hoc use.

## Why

Real-world TEI corpora are full of inline encoding that trip up naive tokenizers: words split across line breaks (`trun-<lb/>cation`), inline emphasis (`Th<hi>is is</hi>`), apparatus criticus (`<app><lem>foo</lem><rdg>bar</rdg></app>`), expansions (`<choice><abbr>Mr.</abbr><expan>Mister</expan></choice>`), table cells, footnotes, multi-word tokens like French `aux → à + les`, and so on. Most tokenizers either lose the inline markup, mangle the structure, or produce output that's not well-formed XML.

xmltokenizer keeps everything:

- Every original element survives the fold (possibly split into `@rpt`/`@cont` continuation fragments).
- Every character of running text is preserved exactly.
- Truncations are joined for the NLP pipeline and restored in the output.
- Multi-word tokens are represented as a surface `<tok>` with `<dtok>` empty children carrying the morphology.
- Sentences (`<s>`) and tokens (`<tok>`) follow TEITOK conventions (`xml:id="w-1"`, `xml:id="s-1"`, …).

## Three usage modes (all first-class)

### 1. As a library — the efficient path (e.g. from flexipipe)

```python
import xmltokenizer as xt

profile  = xt.load_profile("tei")
metadata = xt.extract(input_bytes, profile)

w_counter = [0]   # global token-id counter   (<tok xml:id="w-1">, ...)
s_counter = [0]   # global sentence-id counter (<s   xml:id="s-1">, ...)

for root in metadata.scope_roots:
    xt.build_nlp_plaintext(root, profile)    # inserts \n\n barriers
    plaintext = root.nlp_plaintext           # send THIS to your NLP

    conllu_text = your_nlp_pipeline(plaintext)

    xt.attach_conllu(
        root, conllu_text,
        profile=profile,
        w_counter=w_counter,
        s_counter=s_counter,
    )

output_bytes = xt.fold(metadata)
xt.validate(input_bytes, output_bytes, metadata)   # optional self-check
```

xmltokenizer makes no I/O of its own — every byte that leaves originates from the input XML or the CoNLL-U you handed in.

### 2. Standalone CLI with an external NLP tool

```sh
xmltokenize tokenize file.xml \
    --backend-cmd "udpipe model.udpipe --tokenize --tag --parse" \
    --output tokenized.xml \
    --validate
```

The external command reads plaintext on stdin, emits CoNLL-U on stdout. Works with udpipe, flexipipe, custom pipelines, anything that follows the convention.

### 3. Standalone CLI with the built-in naive tokenizer (zero deps)

```sh
xmltokenize tokenize file.xml --output tokenized.xml
```

Default backend is `naive`: pure-Python whitespace + punctuation tokenization. No external tool needed, no install beyond `pip install xmltokenizer`. The morphology fields are left as `_` (CoNLL-U "unset"), but the XML structure is correct.

## Install

xmltokenizer is not on PyPI (yet). Install directly from the git repo:

```sh
# replace <repo-url> with the actual remote (e.g. https://github.com/maartenpt/xmltokenizer.git)
pip install git+<repo-url>

# or, from a local clone:
git clone <repo-url>
cd xmltokenizer
pip install -e .
```

Requires Python ≥ 3.10. On 3.10 the optional `tomli` dependency is installed automatically; 3.11+ uses stdlib `tomllib`.

## Profiles

Profiles describe how to treat a given XML schema. Two are built-in:

- `tei` (default) — knows about `<text>`, `<teiHeader>`, `<lb>`/`<cb>`/`<pb>` break attributes, `<choice>`/`<app>` preferred-branch rules, `<note>` exclusion, etc.
- `generic` — minimal, root-anchored, no exclusions.

Custom profiles are TOML files with `extends = "tei"` and overrides. See `xmltokenizer/profiles_builtin/tei.toml` for the schema.


## License

MIT.
