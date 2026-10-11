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

## Detokenizing

xmltokenizer refuses input that is already tokenized. To retokenize such a file, strip the existing tokenization first:

```sh
xmltokenize detokenize tokenized.xml --output plain.xml --tsv plain.tokens.tsv
```

This removes `<tok>`, `<dtok>` and `<s>` inside `<text>` (`--tags`, `--scope`), keeping their content, and merges elements that the tokenization had split into fragments (`@rpt`/`@cont`) back into one. All other bytes stay as they were. The TSV keeps a record of everything removed: one row per sentence, token and sub-token, with its id, enclosing sentence and token, character offsets into the text of the scope, its text, and every attribute it carried.

TEI `<w>` tokens are never stripped: a TEI tokenization is converted, not discarded, and `tokenize` refuses files that contain it.

## Span layers: new elements around existing ones

Besides tokenizing, xmltokenizer folds further layers into a document that already has its tokens: each span names its first and last element by id, and one new element goes around them, nested by the same rules as everything else. Typical uses are constituency trees (`<node>` over tokens), named entities, or error and discourse annotation from a standoff source.

```sh
xmltokenize spans tokenized.xml --profile teitok --spans nodes.jsonl \
    --mark-widened widened --output with-nodes.xml
```

`nodes.jsonl` has one span per line: `{"start": "w-2", "end": "w-5", "tag": "node", "attrs": {"cat": "NP"}, "id": "n7", "parent": "n3"}` (`tag`, `attrs`, `id` and `parent` are optional). As a library:

```python
from xmltokenizer import extract, fold, load_profile, Span, attach_spans

profile = load_profile("teitok")
metadata = extract(raw, profile, tokenized=True)
report = attach_spans(metadata.scope_roots[0], [
    Span("w-2", "w-5", tag="node", attrs={"cat": "IP"}, id="n1"),
    Span("w-2", "w-2", tag="node", attrs={"cat": "NP"}, id="n2", parent="n1"),
], profile=profile, mark_widened="widened")
out = fold(metadata)
print(report.summary())
```

What it does:

- The document is read with `tokenized=True` and otherwise left exactly as it is: without the new elements, the output is byte-identical to the input (no whitespace moved, source milestones kept in their order).
- A `<tok>` is never split: a span that starts or ends at a part of a token (a `<dtok>` of a contraction such as *chamam-se* or *da*) covers the whole token, and with `--mark-widened` carries that attribute.
- Spans of one layer nest or stay apart. Two neighbours that would both cover one contraction (one ends in its first part, the next starts in its second) cannot both nest: the later one is left out.
- A span that crosses a source element (`<s>`, `<u>`, `<p>`, an inline element) is left out; `--on-cross split` lets the folder split it instead (fragments with `@rpt`/`@cont`).
- Spans of the same extent nest by `parent`; a span over a whole `<s>` goes inside it; a span over one word wraps the `<tok>`.
- Unknown ids and every left-out span are counted in the report, never guessed.

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
