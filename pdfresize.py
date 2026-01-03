from pypdf import PdfReader, PdfWriter, Transformation

LETTER_W = 612  # 8.5 * 72
LETTER_H = 792  # 11 * 72

reader = PdfReader("DuarteBootOrder.pdf")
writer = PdfWriter()

for page in reader.pages:
    w = float(page.mediabox.width)
    h = float(page.mediabox.height)

    scale = min(LETTER_W / w, LETTER_H / h)

    transform = (
        Transformation()
        .scale(scale)
        .translate((LETTER_W - w * scale) / 2, (LETTER_H - h * scale) / 2)
    )

    page.add_transformation(transform)
    page.mediabox.lower_left = (0, 0)
    page.mediabox.upper_right = (LETTER_W, LETTER_H)

    writer.add_page(page)

with open("DuarteBootOrder_letter.pdf", "wb") as f:
    writer.write(f)

print("All pages normalized to Letter size.")
