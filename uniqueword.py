#!/usr/bin/env python3
"""
pdf_wordlist.py
Extract unique words (case-sensitive) and frequencies from a PDF book.
Also outputs a list of proper nouns.

Outputs (in --outdir):
  - words_freq.tsv        (tab-separated: word<TAB>count, sorted by count desc)
  - unique_words.txt      (unique words only, sorted A→Z)
  - proper_nouns.txt      (unique proper nouns, sorted A→Z)

Proper-noun detection:
  - If spaCy (en_core_web_sm) is available: uses POS tag 'PROPN'
  - Else: heuristic = appears Capitalized at least once and NEVER lowercase

Usage:
  python pdf_wordlist.py input.pdf [--outdir ./output]
"""

import argparse
import os
import re
import sys
import unicodedata
from collections import Counter

# ---- PDF text extraction (pdfminer.six) ----
try:
    from pdfminer.high_level import extract_text
except ImportError:
    sys.stderr.write(
        "Error: pdfminer.six not installed. Install with:\n"
        "  pip install pdfminer.six\n"
    )
    sys.exit(1)

def normalize_unicode(s: str) -> str:
    # Normalize ligatures (ﬁ → fi) and compatibility characters
    return unicodedata.normalize("NFKC", s)

# Token pattern:
#  - letters only, but allow internal apostrophes or hyphens between letter groups
#  - case-sensitive (do NOT lower)
WORD_RE = re.compile(r"[A-Za-z]+(?:['’\-][A-Za-z]+)*")

def extract_words(text: str):
    # Return list of tokens as they appear (case preserved)
    return WORD_RE.findall(text)

def load_spacy_nlp():
    """Try to load spaCy English model for PROPN tagging. Return nlp or None."""
    try:
        import spacy
        try:
            return spacy.load("en_core_web_sm", disable=["ner","lemmatizer","textcat"])
        except OSError:
            # Model not installed
            return None
    except ImportError:
        return None

def find_proper_nouns_spacy(nlp, text: str):
    """
    Use spaCy to find surface-form tokens tagged PROPN.
    Returns a set of unique strings exactly as in the text (case preserved).
    """
    proper = set()
    # Process in chunks to avoid huge docs for long books
    # Split on double newlines / page breaks as crude chunking
    chunks = re.split(r"\n{2,}|\f", text)
    for chunk in chunks:
        if not chunk.strip():
            continue
        doc = nlp(chunk)
        for tok in doc:
            # Keep same tokenization boundary as our regex filter:
            if tok.pos_ == "PROPN" and WORD_RE.fullmatch(tok.text):
                proper.add(tok.text)
    return proper

def find_proper_nouns_heuristic(tokens):
    """
    Heuristic fallback:
    Proper nouns are words that appear capitalized at least once (A...Z start)
    and NEVER appear fully lowercase.
    """
    capitalized = set()
    lowercase = set()
    for w in tokens:
        if w and w[0].isalpha():
            if w[0].isupper():
                capitalized.add(w)
                # Track lowercase version to compare forms
                lowercase.add(w.lower())  # we only need set of forms seen at all
            else:
                lowercase.add(w)
                lowercase.add(w.lower())
    # Build a mapping: original surface forms → lowered
    # We want words that have at least one capitalized surface form
    # and no lowercase surface form (same letters) ever occurred.
    # Compare by lowercased spelling.
    lower_seen = set([t.lower() for t in tokens])
    result = set()
    # A word is candidate if it starts uppercase AND the same lowercase spelling never occurs
    # Note: we must check using lower() of the surface form
    for w in capitalized:
        if w.lower() not in lower_seen - {w.lower()}:
            # Now ensure there wasn't an actual lowercase surface-form of the same string
            # Make a scan for exact lowercase form present:
            # Build lowercase variants set from tokens:
            pass
    # Simpler and correct approach:
    by_lower = {}
    for w in tokens:
        by_lower.setdefault(w.lower(), set()).add(w)
    for lower_form, surfaces in by_lower.items():
        # Is there at least one surface with initial uppercase?
        has_capitalized = any(s[0].isupper() for s in surfaces)
        has_lowercase = any(s[0].islower() for s in surfaces)
        if has_capitalized and not has_lowercase:
            # Add all capitalized surface forms (preserve their exact casing)
            for s in surfaces:
                if s[0].isupper():
                    result.add(s)
    return result

def main():
    ap = argparse.ArgumentParser(description="Extract unique words and proper nouns from a PDF.")
    ap.add_argument("pdf", help="Path to input PDF")
    ap.add_argument("--outdir", default=".", help="Directory for outputs (default: current dir)")
    args = ap.parse_args()

    if not os.path.isfile(args.pdf):
        sys.stderr.write(f"Input not found: {args.pdf}\n")
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)

    # 1) Extract text
    text = extract_text(args.pdf)
    text = normalize_unicode(text)

    # 2) Tokenize (case-sensitive)
    tokens = extract_words(text)

    # 3) Count frequencies
    freq = Counter(tokens)

    # 4) Proper nouns (spaCy if available, else heuristic)
    nlp = load_spacy_nlp()
    if nlp is not None:
        proper = find_proper_nouns_spacy(nlp, text)
        method = "spaCy (PROPN)"
    else:
        proper = find_proper_nouns_heuristic(tokens)
        method = "heuristic"

    # 5) Write outputs
    words_freq_path = os.path.join(args.outdir, "words_freq.tsv")
    unique_words_path = os.path.join(args.outdir, "unique_words.txt")
    proper_nouns_path = os.path.join(args.outdir, "proper_nouns.txt")

    # a) Frequencies TSV (sorted by count desc, then alpha)
    with open(words_freq_path, "w", encoding="utf-8") as f:
        for w, c in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0])):
            f.write(f"{w}\t{c}\n")

    # b) Unique words (alpha)
    with open(unique_words_path, "w", encoding="utf-8") as f:
        for w in sorted(freq.keys()):
            f.write(w + "\n")

    # c) Proper nouns (alpha)
    with open(proper_nouns_path, "w", encoding="utf-8") as f:
        # Keep only those proper nouns that are also in our token set (some spaCy tokens may contain punctuation we filtered)
        proper_in_vocab = sorted(set([p for p in proper if p in freq]))
        for p in proper_in_vocab:
            f.write(p + "\n")

    print(f"Done.\n"
          f"- Words+freq: {words_freq_path}\n"
          f"- Unique words: {unique_words_path}\n"
          f"- Proper nouns ({method}): {proper_nouns_path}")

if __name__ == "__main__":
    main()
