"""Thin command-line wrapper around the library API.

This is a *convenience*. The intended consumer of xmltokenizer is
flexipipe, which uses the library API directly. The CLI exists for
smoke-testing and ad hoc use.

Subcommands:

    extract  INPUT.xml [--plaintext OUT.txt] [--standoff OUT.jsonl]
        Run Phase A and dump the plaintext + xml-layer standoff.

    fold     INPUT.xml --conllu OUT.conllu [--output OUT.xml]
        Run Phase A on INPUT, attach the supplied CoNLL-U, run Phase C.

    tokenize INPUT.xml [--output OUT.xml]
                       [--backend naive|flexipipe|udpipe1-local]
                       [--backend-cmd 'EXEC ARG1 ARG2 ...']
                       [--model PATH]
                       [--profile tei]
                       [--restore-xmlns]
        End-to-end. Default backend: naive (pure-Python; works with no
        external tool installed). Use --backend flexipipe or
        --backend-cmd '...' to wire in a real NLP pipeline.

Run `python -m xmltokenizer <subcommand> --help` for full options.
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

from . import attach_conllu, extract, fold, load_profile
from .backends import (
    BackendError,
    ExternalCommandBackend,
    NaiveBackend,
    flexipipe_backend,
    udpipe1_backend,
)
from .namespace import deactivate, reactivate
from .orchestrator import run as run_pipeline
from .records import write_jsonl
from .validate import ValidationError, run as validate_run


# ---------------------------------------------------------------------
# Subcommand: extract
# ---------------------------------------------------------------------


def cmd_extract(args: argparse.Namespace) -> int:
    raw = Path(args.input).read_bytes()
    raw_de, _ = deactivate(raw)
    profile = load_profile(args.profile)
    metadata = extract(raw_de, profile, source_path=args.input)

    # Write plaintext (concatenated across scope roots, separated by a
    # form-feed character for clarity).
    if args.plaintext:
        Path(args.plaintext).write_text(
            "\x0c".join(r.fold_plaintext for r in metadata.scope_roots),
            encoding="utf-8",
        )
    else:
        for r in metadata.scope_roots:
            sys.stdout.write(r.fold_plaintext)

    # Optional: write all xml-layer records as JSONL.
    if args.standoff:
        all_recs = []
        for r in metadata.scope_roots:
            all_recs.extend(r.xml_layer.records)
        write_jsonl(args.standoff, all_recs)

    return 0


# ---------------------------------------------------------------------
# Subcommand: fold
# ---------------------------------------------------------------------


def cmd_fold(args: argparse.Namespace) -> int:
    raw = Path(args.input).read_bytes()
    raw_de, ns_transform = deactivate(raw)
    profile = load_profile(args.profile)
    metadata = extract(raw_de, profile, source_path=args.input)

    conllu_text = Path(args.conllu).read_text(encoding="utf-8")

    # For multi-scope-root inputs, the simplest contract is "all CoNLL-U
    # in one stream applies to the first non-empty scope root". A future
    # version may support per-scope CoNLL-U files.
    w_counter = [0]
    s_counter = [0]
    for r in metadata.scope_roots:
        if not r.fold_plaintext.strip():
            continue
        attach_conllu(
            r, conllu_text,
            profile=profile, w_counter=w_counter, s_counter=s_counter,
        )
        break

    out = fold(metadata)
    if args.restore_xmlns and ns_transform is not None:
        out = reactivate(out, ns_transform)

    if args.output:
        Path(args.output).write_bytes(out)
    else:
        sys.stdout.buffer.write(out)
    return 0


# ---------------------------------------------------------------------
# Subcommand: tokenize (end-to-end)
# ---------------------------------------------------------------------


def cmd_tokenize(args: argparse.Namespace) -> int:
    backend = _build_backend(args)
    raw = Path(args.input).read_bytes()
    # Build the pipeline ourselves so we can hand the metadata to the
    # validator. Mirrors xmltokenizer.orchestrator.run.
    profile = load_profile(args.profile)
    raw_de, ns_transform = deactivate(raw)
    metadata = extract(raw_de, profile, source_path=args.input)
    from . import build_nlp_plaintext as _build_nlp
    w_counter = [0]
    s_counter = [0]
    for root in metadata.scope_roots:
        if not root.fold_plaintext.strip():
            continue
        _build_nlp(root, profile)
        conllu_text = backend.tokenize(root.nlp_plaintext)
        attach_conllu(
            root, conllu_text,
            profile=profile, w_counter=w_counter, s_counter=s_counter,
        )
    out = fold(metadata)
    if args.restore_xmlns and ns_transform is not None:
        out = reactivate(out, ns_transform)
    if args.validate:
        try:
            ran = validate_run(raw, out, metadata)
            print(f"# validation passed: {', '.join(ran)}", file=sys.stderr)
        except ValidationError as e:
            print(f"validation error: {e}", file=sys.stderr)
            return 3
    if args.output:
        Path(args.output).write_bytes(out)
    else:
        sys.stdout.buffer.write(out)
    return 0


def _build_backend(args: argparse.Namespace):
    if args.backend_cmd:
        argv = shlex.split(args.backend_cmd)
        return ExternalCommandBackend(argv=argv)
    if args.backend == "naive":
        return NaiveBackend()
    if args.backend == "flexipipe":
        kwargs = {}
        if args.flexipipe_tasks:
            kwargs["tasks"] = args.flexipipe_tasks
        return flexipipe_backend(**kwargs)
    if args.backend == "udpipe1-local":
        if not args.model:
            raise SystemExit("--model is required when --backend=udpipe1-local")
        return udpipe1_backend(args.model)
    raise SystemExit(f"unknown backend: {args.backend!r}")


# ---------------------------------------------------------------------
# Argparse plumbing
# ---------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="xmltokenize",
        description=(
            "Inline tokenization of TEI/XML. Primarily a library "
            "(see `import xmltokenizer`); this CLI is a convenience."
        ),
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # --- extract ---
    p_ex = sub.add_parser("extract", help="Run Phase A: emit plaintext + standoff.")
    p_ex.add_argument("input")
    p_ex.add_argument("--profile", default="tei")
    p_ex.add_argument("--plaintext", help="Write plaintext here (default: stdout)")
    p_ex.add_argument("--standoff", help="Write xml-layer records as JSONL")
    p_ex.set_defaults(func=cmd_extract)

    # --- fold ---
    p_fo = sub.add_parser(
        "fold",
        help="Phase A + attach external CoNLL-U + Phase C.",
    )
    p_fo.add_argument("input")
    p_fo.add_argument("--conllu", required=True, help="Path to CoNLL-U file")
    p_fo.add_argument("--profile", default="tei")
    p_fo.add_argument("--output", help="Write XML here (default: stdout)")
    p_fo.add_argument("--restore-xmlns", action="store_true")
    p_fo.set_defaults(func=cmd_fold)

    # --- tokenize ---
    p_tok = sub.add_parser(
        "tokenize",
        help="End-to-end: extract → backend → fold. Default backend: flexipipe.",
    )
    p_tok.add_argument("input")
    p_tok.add_argument("--profile", default="tei")
    p_tok.add_argument("--output", help="Write XML here (default: stdout)")
    p_tok.add_argument(
        "--backend",
        choices=["naive", "flexipipe", "udpipe1-local"],
        default="naive",
        help=(
            "Backend name (default: naive — built-in whitespace + punctuation "
            "tokenizer, zero deps). Use 'flexipipe' or 'udpipe1-local' for "
            "real tagging/parsing. Ignored if --backend-cmd is given."
        ),
    )
    p_tok.add_argument(
        "--backend-cmd",
        help=(
            "Override: full shell-quoted command line to run. The command "
            "reads plaintext on stdin and writes CoNLL-U on stdout."
        ),
    )
    p_tok.add_argument("--model", help="Model path (for udpipe1-local)")
    p_tok.add_argument(
        "--flexipipe-tasks",
        help="Task list passed to flexipipe (--tasks=...). Default: tokenize,tag,parse",
    )
    p_tok.add_argument("--restore-xmlns", action="store_true")
    p_tok.add_argument(
        "--validate",
        action="store_true",
        help=(
            "Run V1, V4, V5, V6, V7 invariants on the output and exit "
            "non-zero on any failure. Adds ~5-10%% runtime."
        ),
    )
    p_tok.set_defaults(func=cmd_tokenize)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except BackendError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
