from pydantic import BaseModel


class ValidationCheck(BaseModel):
    id: str
    name: str
    status: str
    detail: str
    value: dict | list | str | int | float | bool | None


class ValidationResponse(BaseModel):
    status: str
    overall_status: str
    paper_id: int
    paper_title: str
    checks: list[ValidationCheck]
    metrics: dict
    metadata_fields: dict
    issues: list[str]
    warnings: list[str]
