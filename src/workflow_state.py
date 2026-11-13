"""Shared state keys used by the factory's implemented and future nodes."""
from typing import Any, TypedDict


class FactoryState(TypedDict, total=False):
    model_id: str
    inventory_record: dict
    input_root: str
    doc_path: str | None
    ir_path: str | None
    generated_code_path: str | None
    parity_result: dict | None
    retry_count: int
    metrics_path: str | None
    report_path: str | None
    approvals: list[dict]
    status: str


def shared_state(values: dict[str, Any]) -> dict[str, Any]:
    return {"doc_path": None, "ir_path": None, "generated_code_path": None,
            "parity_result": None, "retry_count": 0, "metrics_path": None,
            "report_path": None, "approvals": [], **values}
