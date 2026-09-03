from app.services.parsing.pdf_parser import extract_pdf_text


PDF_PATH = r"D:\FYP\main\storage\papers\c44fba57401d45e6818443f27e7caf7f.pdf"


result = extract_pdf_text(PDF_PATH)


print("Page count:")
print(result["page_count"])

print("\nNeeds OCR:")
print(result["needs_ocr"])

print("\nExtracted text length:")
print(len(result["full_text"]))

print("\nFirst 1000 characters:")
print(result["full_text"][:1000])