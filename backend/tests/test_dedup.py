#D:\FYP\main\backend\test_dedup.py


from app.services.dedup.title_normalizer import normalize_title


titles = [
    "Attention Is All You Need",
    "attention is all you need"
    "Attention-Is-All-You-Need!"
]

for title in titles:
    print(title,
          "->",
          normalize_title(title)
    )


{
  "title": "attention is all you need",
  "source": "openalex",
  "source_id": "different-id"
}    