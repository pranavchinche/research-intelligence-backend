from app.services.parsing.pdf_parser import extract_pdf_text
from app.services.parsing.chunker import chunk_text
from app.services.embeddings.embedder import generate_embeddings


PDF_PATH = r"D:\FYP\main\storage\papers\c44fba57401d45e6818443f27e7caf7f.pdf"


parsed = extract_pdf_text(PDF_PATH)

if parsed["needs_ocr"]:
    print("PDF requires OCR.")
    raise SystemExit

chunks = chunk_text(
    parsed["pages"]
)

texts = [
    chunk["text"]
    for chunk in chunks[:5]
]

embeddings = generate_embeddings(
    texts
)

print("Chunks:", len(texts))
print("Embeddings:", len(embeddings))

if embeddings:
    print("Vector dimensions:", len(embeddings[0]))
    print("First vector:")
    print(embeddings[0][:10])