"""Bounded, lossless source windows with original local line numbers.

Stage labels are routing hints, not a SAS/R parser or a proof of semantics.
"""
from __future__ import annotations

import re
from discover_agent import Record

STAGES = {
    "data": r"\b(?:read[._]csv|read_csv|read_table|read\.table|read_excel|read_sas|fread|load|proc\s+import|infile|input|data)\b",
    "variables": r"\b(?:mutate|ifelse|fillna|impute|replace_na|cut|qcut|clip|median|mean|transform|encode|dropna|merge|join)\b",
    "fit": r"\b(?:(?:lightgbm|lgb\.train|xgboost|xgb\.train|glm|lm|train|fit)\s*\(|proc\s+(?:logistic|reg|hpforest|hpsplit)\b)",
    "scoring": r"\b(?:predict|predict_proba|score|scoring|proc\s+score|output\s+out)\b",
}


def chunk_sources(record: Record, max_characters: int = 12000) -> list[dict]:
    """Keep every source line exactly once, splitting at stage or size boundaries."""
    if max_characters < 256:
        raise ValueError("Chunk budget must be at least 256 characters")
    chunks = []
    for source in record.sources:
        start, pending, size = 1, [], 0
        stage = "context" if source.kind != "code" else "other"

        def flush():
            if pending:
                chunks.append({"chunk_id": f"{source.source_id}:L{start}-{start + len(pending) - 1}",
                               "source_id": source.source_id, "kind": source.kind, "stage": stage,
                               "line_start": start, "line_end": start + len(pending) - 1,
                               "lines": [f"{start + i}: {line}" for i, line in enumerate(pending)]})

        for number, line in enumerate(source.text.splitlines(), 1):
            cost = len(line) + len(str(number)) + 3
            if cost > max_characters:
                raise ValueError(f"Source line exceeds chunk budget: {source.source_id}:L{number}")
            next_stage = stage
            stripped = line.strip()
            if source.kind == "code" and not stripped.startswith(("#", "//", "*")):
                # Scoring takes precedence over a generic data/transform keyword.
                for candidate in ("scoring", "fit", "variables", "data"):
                    if re.search(STAGES[candidate], line, re.I):
                        next_stage = candidate
                        break
            if pending and (next_stage != stage or size + cost > max_characters):
                flush()
                pending, size, start = [], 0, number
            if not pending:
                start = number
            stage = next_stage
            pending.append(line)
            size += cost
        flush()
    return chunks


def group_chunks(chunks: list[dict], max_characters: int = 40000) -> list[list[dict]]:
    """Pack lossless source windows so short models retain their full context."""
    import json
    groups, group, size = [], [], 2
    for chunk in chunks:
        cost = len(json.dumps(chunk, ensure_ascii=False)) + 2
        if group and size + cost > max_characters:
            groups.append(group)
            group, size = [], 2
        if cost + 2 > max_characters:
            raise ValueError("A source window exceeds the group budget; reduce --chunk-characters")
        group.append(chunk)
        size += cost
    if group:
        groups.append(group)
    return groups


def refs_within_group(evidence, group):
    """Accept a citation only when every original line was supplied in this group."""
    for ref in evidence:
        end = ref.line_start
        for chunk in sorted((c for c in group if c['source_id'] == ref.source_id),
                            key=lambda c: c['line_start']):
            if chunk['line_start'] <= end <= chunk['line_end']:
                end = chunk['line_end'] + 1
        if end <= ref.line_end:
            raise ValueError("Evidence must cite only the supplied source group")
