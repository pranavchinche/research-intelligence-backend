from app.services.parsing.pdf_parser import extract_pdf_text
from app.services.parsing.chunker import chunk_text


PDF_PATH = r"D:\FYP\main\storage\papers\c44fba57401d45e6818443f27e7caf7f.pdf"


parsed = extract_pdf_text(PDF_PATH)

print("Page count:", parsed["page_count"])
print("Needs OCR:", parsed["needs_ocr"])
print("Text length:", len(parsed["full_text"]))


if parsed["needs_ocr"]:
    print("\nPDF requires OCR.")

else:
    chunks = chunk_text(
        parsed["pages"]
    )

    print("\nTotal chunks:")
    print(len(chunks))

    print("\nFirst chunk:")
    print(chunks[0])

    print("\nLast chunk:")
    print(chunks[-1])