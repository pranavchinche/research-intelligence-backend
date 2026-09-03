def validate_paper(
    paper: dict,
) -> tuple[bool, list[str]]:

    errors = []

    # -------------------------
    # TITLE
    # -------------------------
    title = paper.get("title")

    if not title or not str(title).strip():
        errors.append("Paper title is missing")

    # -------------------------
    # SOURCE
    # -------------------------
    source = paper.get("source")

    if not source or not str(source).strip():
        errors.append("Paper source is missing")

    # -------------------------
    # AUTHORS
    # -------------------------
    authors = paper.get("authors")

    if authors is None:
        errors.append("Paper authors are missing")

    elif not isinstance(authors, list):
        errors.append("Paper authors must be a list")

    # -------------------------
    # PDF
    # -------------------------
    pdf_url = paper.get("pdf_url")
    pdf_path = paper.get("pdf_path")

    # A paper can exist without a PDF at this stage.
    # Do not reject it just because PDF is unavailable.

    # -------------------------
    # RESULT
    # -------------------------
    return (
        len(errors) == 0,
        errors,
    )