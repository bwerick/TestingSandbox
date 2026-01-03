#!/usr/bin/env python3

import argparse
from pathlib import Path
from pypdf import PdfWriter, PdfReader


def merge_pdfs(input_files, output_pdf):
    writer = PdfWriter()

    for pdf_path in input_files:
        print(f"Adding: {pdf_path}")
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            writer.add_page(page)

    with open(output_pdf, "wb") as f:
        writer.write(f)

    print(f"\nMerged PDF saved to: {output_pdf}")


def main():
    parser = argparse.ArgumentParser(description="Merge multiple PDFs into one.")
    parser.add_argument(
        "pdfs",
        nargs="+",
        help="Input PDF files (supports glob expansion by the shell)",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output merged PDF filename",
    )

    args = parser.parse_args()

    input_files = [Path(p) for p in args.pdfs]

    merge_pdfs(input_files, args.output)


if __name__ == "__main__":
    main()
