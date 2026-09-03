#D:\FYP\main\backend\test_normalization.py

from app.services.validation.normalize import normalize_paper

from app.services.validation.paper_validator import validate_paper


paper = {
    "title": "Attention IS All you Need",
    "abstract": "Transformer architecture paper.",
    "authors": "Vaswani, shezeer, parmar",
    "doi": None,
    "source_id": "test-001",
    "arvix_id": "1706.03762",
    "published_date": "2012-06-12",
    "updated_date": None,
    "categories": "cs.AI, cs.CL",
    "pdf_url": "https://arxiv.org/pdf/1706.03762.pdf",
}

normalized = normalize_paper(
    paper,
    "arvix"
)

valid, errors = validate_paper(normalized)

print("Normalized:")
print(normalized)

print("\nValid:")
print(valid)

print("\nErrors:")
print(errors)