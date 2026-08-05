from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from tools import (
    count_pdf_pages,
    extract_pdf_text,
    find_pdf_files,
    make_asset_descriptor,
    pdf_first_page_preview,
    pdf_info,
)


def ensure_dir(path: str | Path) -> Path:
    target = Path(path).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    return target


def write_text(path: str | Path, content: object) -> str:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("" if content is None else str(content), encoding="utf-8")
    return str(target)


def write_json(path: str | Path, content: object) -> str:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(target)


def excerpt(text: object, limit: int = 6000) -> str:
    raw = "" if text is None else str(text)
    return raw if len(raw) <= limit else raw[:limit] + "\n...(truncated)..."


def parse_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_json_block(text: object) -> Any:
    raw = str(text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        raw = fenced.group(1).strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    first_obj = raw.find("{")
    last_obj = raw.rfind("}")
    if first_obj != -1 and last_obj != -1 and last_obj > first_obj:
        return json.loads(raw[first_obj : last_obj + 1])

    first_arr = raw.find("[")
    last_arr = raw.rfind("]")
    if first_arr != -1 and last_arr != -1 and last_arr > first_arr:
        return json.loads(raw[first_arr : last_arr + 1])

    raise ValueError("No valid JSON block found in agent output.")


def _extract_json_array(text: object, *, preferred_keys: list[str]) -> list[dict[str, object]]:
    parsed = _extract_json_block(text)
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    if isinstance(parsed, dict):
        for key in preferred_keys:
            value = parsed.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
            if isinstance(value, str):
                return _extract_json_array(value, preferred_keys=[])
        for value in parsed.values():
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
            if isinstance(value, str):
                try:
                    return _extract_json_array(value, preferred_keys=[])
                except Exception:
                    pass
    raise ValueError("Agent output did not contain a JSON array.")


def _strip_closing_tag(text: object, tag_name: str) -> str:
    raw = str(text or "").strip()
    return re.sub(rf"</\s*{re.escape(tag_name)}\s*>", "", raw, flags=re.IGNORECASE).strip()


def _fallback_summary_record(paper: dict[str, object], error_text: str = "") -> dict[str, object]:
    return {
        "paper_id": paper.get("paper_id", ""),
        "title": paper.get("title_guess") or paper.get("filename") or "",
        "one_sentence_summary": "Summary generation failed.",
        "problem": "",
        "method": "",
        "innovation_points": [],
        "results": [],
        "limitations": [error_text] if error_text else [],
        "why_it_matters": "",
        "keywords": [],
        "paper_type": "unknown",
    }


def _fallback_score_record(paper_id: str, agent_name: str, error_text: str = "") -> dict[str, object]:
    return {
        "paper_id": paper_id,
        "agent": agent_name,
        "status": "error",
        "score": 0.0,
        "confidence": 0.0,
        "strengths": [],
        "concerns": [error_text] if error_text else [],
        "reasoning": "Scoring failed." if error_text else "",
        "is_incremental_ab": False,
    }


def init_daily_digest_run(_: dict[str, object], attrs: dict[str, object]) -> dict[str, object]:
    pdf_dir = Path(str(attrs.get("pdf_dir") or "")).expanduser().resolve()
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory does not exist: {pdf_dir}")

    run_name = str(attrs.get("run_name") or f"digest_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    output_root = ensure_dir(attrs.get("output_dir") or (Path.cwd() / "runs"))
    run_dir = ensure_dir(output_root / run_name)

    source_dir = ensure_dir(run_dir / "00_sources")
    extract_dir = ensure_dir(run_dir / "01_extracted")
    summary_dir = ensure_dir(run_dir / "02_summaries")
    score_dir = ensure_dir(run_dir / "03_scores")
    brief_dir = ensure_dir(run_dir / "04_brief")
    tool_asset_root = ensure_dir(run_dir / "05_tool_assets")

    date_label = str(attrs.get("date_label") or datetime.now().strftime("%Y-%m-%d"))
    max_papers = int(attrs.get("max_papers") or 0)

    return {
        "pdf_dir": str(pdf_dir),
        "run_name": run_name,
        "run_dir": str(run_dir),
        "source_dir": str(source_dir),
        "extract_dir": str(extract_dir),
        "summary_dir": str(summary_dir),
        "score_dir": str(score_dir),
        "brief_dir": str(brief_dir),
        "tool_asset_root": str(tool_asset_root),
        "date_label": date_label,
        "max_papers": max_papers,
        "paper_inventory_path": str(source_dir / "paper_inventory.json"),
        "paper_summaries_path": str(summary_dir / "paper_summaries.json"),
        "novelty_scores_path": str(score_dir / "novelty_scores.json"),
        "rigor_scores_path": str(score_dir / "rigor_scores.json"),
        "impact_scores_path": str(score_dir / "impact_scores.json"),
        "aggregated_scores_path": str(score_dir / "aggregated_scores.json"),
        "daily_brief_json_path": str(brief_dir / "daily_brief.json"),
        "daily_brief_markdown_path": str(brief_dir / "daily_brief.md"),
    }


def extract_pdf_corpus(_: dict[str, object], attrs: dict[str, object]) -> dict[str, object]:
    pdf_paths = find_pdf_files(attrs["pdf_dir"])
    max_papers = int(attrs.get("max_papers") or 0)
    if max_papers > 0:
        pdf_paths = pdf_paths[:max_papers]
    if not pdf_paths:
        raise RuntimeError(f"No PDF files found under {attrs['pdf_dir']}")

    extract_dir = ensure_dir(attrs["extract_dir"])
    preview_dir = ensure_dir(extract_dir / "previews")
    text_dir = ensure_dir(extract_dir / "texts")
    papers: list[dict[str, object]] = []
    pdf_assets: list[dict[str, str]] = []
    preview_assets: list[dict[str, str]] = []

    for index, pdf_path in enumerate(pdf_paths, start=1):
        info = pdf_info(pdf_path)
        text_path = text_dir / f"{index:03d}_{pdf_path.stem}.txt"
        text = extract_pdf_text(pdf_path, text_path)
        preview_path = pdf_first_page_preview(pdf_path, preview_dir / f"{index:03d}_{pdf_path.stem}")
        paper_id = f"paper_{index:03d}"
        pdf_asset = make_asset_descriptor(pdf_path, "pdf", f"{paper_id} PDF", paper_id=paper_id)
        preview_asset = make_asset_descriptor(preview_path, "image", f"{paper_id} preview", paper_id=paper_id)
        papers.append(
            {
                "paper_id": paper_id,
                "filename": pdf_path.name,
                "pdf_path": str(pdf_path),
                "text_path": str(text_path),
                "preview_path": preview_path,
                "title_guess": info.get("Title") or pdf_path.stem,
                "page_count": count_pdf_pages(pdf_path),
                "text_excerpt": excerpt(text, limit=9000),
                "full_text_char_count": len(text),
                "pdf_asset": pdf_asset,
                "preview_image_asset": preview_asset,
            }
        )
        pdf_assets.append(pdf_asset)
        preview_assets.append(preview_asset)

    inventory_path = write_json(attrs["paper_inventory_path"], papers)
    return {
        "paper_count": len(papers),
        "paper_inventory": papers,
        "paper_inventory_path": inventory_path,
        "paper_inventory_excerpt": excerpt(json.dumps(papers, ensure_ascii=False, indent=2), limit=12000),
        "paper_pdf_assets": pdf_assets,
        "paper_preview_images": preview_assets,
    }


def _chunk_records(records: list[dict[str, object]], batch_size: int) -> list[list[dict[str, object]]]:
    safe_size = max(1, int(batch_size or len(records) or 1))
    return [records[index : index + safe_size] for index in range(0, len(records), safe_size)]


def _build_summary_payload(batch: list[dict[str, object]], tool_asset_root: str) -> list[dict[str, object]]:
    return [
        {
            "paper_id": item["paper_id"],
            "filename": item["filename"],
            "title_guess": item["title_guess"],
            "pdf_path": item["pdf_path"],
            "pdf_asset": item.get("pdf_asset") or {},
            "page_count": item["page_count"],
            "preview_path": item["preview_path"],
            "preview_image_asset": item.get("preview_image_asset") or {},
            "text_excerpt": item["text_excerpt"],
            "tool_asset_root": tool_asset_root,
        }
        for item in batch
    ]


def _normalize_summary_record(record: dict[str, object], paper: dict[str, object]) -> dict[str, object]:
    return {
        "paper_id": str(record.get("paper_id") or paper["paper_id"]),
        "title": record.get("title") or paper.get("title_guess") or paper.get("filename") or "",
        "one_sentence_summary": record.get("one_sentence_summary") or "",
        "problem": record.get("problem") or "",
        "method": record.get("method") or "",
        "innovation_points": record.get("innovation_points") or [],
        "results": record.get("results") or [],
        "limitations": record.get("limitations") or [],
        "why_it_matters": record.get("why_it_matters") or "",
        "keywords": record.get("keywords") or [],
        "paper_type": record.get("paper_type") or "unknown",
    }


def _normalize_score_record(record: dict[str, object], paper: dict[str, object], agent_name: str) -> dict[str, object]:
    return {
        "paper_id": str(paper.get("paper_id") or ""),
        "title": (paper.get("summary") or {}).get("title") or paper.get("title_guess") or paper.get("filename"),
        "agent": agent_name,
        "status": "ok",
        "score": round(parse_float(record.get("score")), 2),
        "confidence": round(parse_float(record.get("confidence")), 2),
        "strengths": record.get("strengths") or [],
        "concerns": record.get("concerns") or [],
        "reasoning": record.get("reasoning") or "",
        "is_incremental_ab": bool(record.get("is_incremental_ab")) if agent_name == "novelty" else False,
    }


def prepare_summary_batches(_: dict[str, object], attrs: dict[str, object]) -> dict[str, object]:
    inventory = list(attrs.get("paper_inventory", []))
    batch_size = int(attrs.get("summary_batch_size") or 4)
    batches = _chunk_records(inventory, batch_size)
    return {
        "summary_batches": batches,
        "summary_batch_index": 0,
        "summary_batch_total": len(batches),
        "summary_records": [],
    }


def extract_corpus_and_prepare_summary_batches(_: dict[str, object], attrs: dict[str, object]) -> dict[str, object]:
    extracted = extract_pdf_corpus({}, attrs)
    next_attrs = {**attrs, **extracted}
    prepared = prepare_summary_batches({}, next_attrs)
    return {**extracted, **prepared}


def prepare_current_summary_batch(_: dict[str, object], attrs: dict[str, object]) -> dict[str, object]:
    batches = list(attrs.get("summary_batches", []))
    index = int(attrs.get("summary_batch_index") or 0)
    if index >= len(batches):
        batch: list[dict[str, object]] = []
    else:
        batch = list(batches[index])
    payload = _build_summary_payload(batch, str(attrs.get("tool_asset_root") or ""))
    return {
        "current_summary_batch": batch,
        "paper_inventory_payload": json.dumps(payload, ensure_ascii=False, indent=2),
        "paper_pdf_assets": [item.get("pdf_asset") or {} for item in batch if item.get("pdf_asset")],
        "paper_preview_images": [
            item.get("preview_image_asset") or {}
            for item in batch
            if item.get("preview_image_asset")
        ],
    }


def persist_summary_batch(input_dict: dict[str, object], attrs: dict[str, object]) -> dict[str, object]:
    batch = list(attrs.get("current_summary_batch", []))
    index = int(attrs.get("summary_batch_index") or 0)
    records = list(attrs.get("summary_records", []))
    raw = _strip_closing_tag(
        input_dict.get("paper_summaries_json", "") or attrs.get("paper_summaries_json", ""),
        "paper_summaries_json",
    )
    parse_error = ""
    parsed_by_id: dict[str, dict[str, object]] = {}

    try:
        parsed = _extract_json_array(raw, preferred_keys=["paper_summaries_json", "summaries", "items"])
        parsed_by_id = {
            str(item.get("paper_id") or ""): dict(item)
            for item in parsed
            if isinstance(item, dict) and item.get("paper_id")
        }
    except Exception as exc:
        parse_error = str(exc)

    batch_records: list[dict[str, object]] = []
    for paper in batch:
        paper_dict = dict(paper)
        paper_id = str(paper_dict.get("paper_id") or "")
        record = parsed_by_id.get(paper_id)
        if record is None:
            error_text = parse_error or "Missing summary record from agent."
            paper_dict["summary"] = _fallback_summary_record(paper_dict, error_text)
            paper_dict["summary_status"] = "error"
            paper_dict["summary_error"] = error_text
        else:
            paper_dict["summary"] = _normalize_summary_record(record, paper_dict)
            paper_dict["summary_status"] = "ok"
            paper_dict["summary_error"] = ""
        batch_records.append(paper_dict)

    return {
        "summary_records": records + batch_records,
        "summary_batch_index": index + 1,
    }


def finalize_summary_outputs(_: dict[str, object], attrs: dict[str, object]) -> dict[str, object]:
    inventory = list(attrs.get("paper_inventory", []))
    records_by_id = {
        str(item.get("paper_id") or ""): dict(item)
        for item in attrs.get("summary_records", [])
        if isinstance(item, dict)
    }

    ordered_records: list[dict[str, object]] = []
    for paper in inventory:
        paper_id = str(paper.get("paper_id") or "")
        record = records_by_id.get(paper_id)
        if record is None:
            paper_dict = dict(paper)
            paper_dict["summary"] = _fallback_summary_record(paper_dict, "Summary record missing after batch loop.")
            paper_dict["summary_status"] = "error"
            paper_dict["summary_error"] = "Summary record missing after batch loop."
            ordered_records.append(paper_dict)
        else:
            ordered_records.append(record)

    summaries_path = write_json(attrs["paper_summaries_path"], ordered_records)
    return {
        "paper_summaries": ordered_records,
        "paper_summaries_path": summaries_path,
        "paper_summaries_excerpt": excerpt(json.dumps(ordered_records, ensure_ascii=False, indent=2), limit=15000),
    }


def _build_scoring_payload(batch: list[dict[str, object]]) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    for item in batch:
        summary = item.get("summary") or {}
        payload.append(
            {
                "paper_id": item["paper_id"],
                "title": summary.get("title") or item.get("title_guess"),
                "one_sentence_summary": summary.get("one_sentence_summary") or "",
                "problem": summary.get("problem") or "",
                "method": summary.get("method") or "",
                "innovation_points": summary.get("innovation_points") or [],
                "results": summary.get("results") or [],
                "limitations": summary.get("limitations") or [],
                "why_it_matters": summary.get("why_it_matters") or "",
                "keywords": summary.get("keywords") or [],
                "paper_type": summary.get("paper_type") or "unknown",
            }
        )
    return payload


def _prepare_score_batches(attrs: dict[str, object], agent_name: str) -> dict[str, object]:
    papers = [
        dict(item)
        for item in attrs.get("paper_summaries", [])
        if isinstance(item, dict) and str(item.get("summary_status") or "") == "ok"
    ]
    batch_size = int(attrs.get("scoring_batch_size") or 20)
    return {
        "score_agent_name": agent_name,
        "score_batch_index": 0,
        "score_records": [],
        "scoring_batches": _chunk_records(papers, batch_size),
    }


def prepare_novelty_batches(_: dict[str, object], attrs: dict[str, object]) -> dict[str, object]:
    prepared = _prepare_score_batches(attrs, "novelty")
    return {
        "novelty_batches": prepared["scoring_batches"],
        "novelty_batch_index": prepared["score_batch_index"],
        "novelty_records": prepared["score_records"],
    }


def prepare_rigor_batches(_: dict[str, object], attrs: dict[str, object]) -> dict[str, object]:
    prepared = _prepare_score_batches(attrs, "rigor")
    return {
        "rigor_batches": prepared["scoring_batches"],
        "rigor_batch_index": prepared["score_batch_index"],
        "rigor_records": prepared["score_records"],
    }


def prepare_impact_batches(_: dict[str, object], attrs: dict[str, object]) -> dict[str, object]:
    prepared = _prepare_score_batches(attrs, "impact")
    return {
        "impact_batches": prepared["scoring_batches"],
        "impact_batch_index": prepared["score_batch_index"],
        "impact_records": prepared["score_records"],
    }


def prepare_current_score_batch(_: dict[str, object], attrs: dict[str, object]) -> dict[str, object]:
    batches = list(attrs.get("scoring_batches", []))
    index = int(attrs.get("score_batch_index") or 0)
    if index >= len(batches):
        batch: list[dict[str, object]] = []
    else:
        batch = list(batches[index])
    payload = _build_scoring_payload(batch)
    return {
        "current_scoring_batch": batch,
        "paper_summaries_payload": json.dumps(payload, ensure_ascii=False, indent=2),
    }


def prepare_current_novelty_batch(_: dict[str, object], attrs: dict[str, object]) -> dict[str, object]:
    prepared = prepare_current_score_batch(
        {},
        {
            **attrs,
            "scoring_batches": attrs.get("novelty_batches", []),
            "score_batch_index": attrs.get("novelty_batch_index", 0),
        },
    )
    return {
        "novelty_current_batch": prepared["current_scoring_batch"],
        "paper_summaries_payload": prepared["paper_summaries_payload"],
    }


def prepare_current_rigor_batch(_: dict[str, object], attrs: dict[str, object]) -> dict[str, object]:
    prepared = prepare_current_score_batch(
        {},
        {
            **attrs,
            "scoring_batches": attrs.get("rigor_batches", []),
            "score_batch_index": attrs.get("rigor_batch_index", 0),
        },
    )
    return {
        "rigor_current_batch": prepared["current_scoring_batch"],
        "paper_summaries_payload": prepared["paper_summaries_payload"],
    }


def prepare_current_impact_batch(_: dict[str, object], attrs: dict[str, object]) -> dict[str, object]:
    prepared = prepare_current_score_batch(
        {},
        {
            **attrs,
            "scoring_batches": attrs.get("impact_batches", []),
            "score_batch_index": attrs.get("impact_batch_index", 0),
        },
    )
    return {
        "impact_current_batch": prepared["current_scoring_batch"],
        "paper_summaries_payload": prepared["paper_summaries_payload"],
    }


def persist_score_batch(input_dict: dict[str, object], attrs: dict[str, object]) -> dict[str, object]:
    batch = list(attrs.get("current_scoring_batch", []))
    index = int(attrs.get("score_batch_index") or 0)
    agent_name = str(attrs.get("score_agent_name") or "").strip() or "score"
    records = list(attrs.get("score_records", []))
    raw = _strip_closing_tag(
        input_dict.get(f"{agent_name}_scores_json", "") or attrs.get(f"{agent_name}_scores_json", ""),
        f"{agent_name}_scores_json",
    )
    parse_error = ""
    parsed_by_id: dict[str, dict[str, object]] = {}

    try:
        parsed = _extract_json_array(
            raw,
            preferred_keys=[f"{agent_name}_scores_json", f"{agent_name}_scores", "scores", "items"],
        )
        parsed_by_id = {
            str(item.get("paper_id") or ""): dict(item)
            for item in parsed
            if isinstance(item, dict) and item.get("paper_id")
        }
    except Exception as exc:
        parse_error = str(exc)

    batch_records: list[dict[str, object]] = []
    for paper in batch:
        paper_id = str(paper.get("paper_id") or "")
        record = parsed_by_id.get(paper_id)
        if record is None:
            batch_records.append(_fallback_score_record(paper_id, agent_name, parse_error or "Missing score record from agent."))
        else:
            batch_records.append(_normalize_score_record(record, paper, agent_name))

    return {
        "score_records": records + batch_records,
        "score_batch_index": index + 1,
    }


def persist_novelty_batch(input_dict: dict[str, object], attrs: dict[str, object]) -> dict[str, object]:
    persisted = persist_score_batch(
        input_dict,
        {
            **attrs,
            "current_scoring_batch": attrs.get("novelty_current_batch", []),
            "score_batch_index": attrs.get("novelty_batch_index", 0),
            "score_records": attrs.get("novelty_records", []),
            "score_agent_name": "novelty",
        },
    )
    return {
        "novelty_records": persisted["score_records"],
        "novelty_batch_index": persisted["score_batch_index"],
    }


def persist_rigor_batch(input_dict: dict[str, object], attrs: dict[str, object]) -> dict[str, object]:
    persisted = persist_score_batch(
        input_dict,
        {
            **attrs,
            "current_scoring_batch": attrs.get("rigor_current_batch", []),
            "score_batch_index": attrs.get("rigor_batch_index", 0),
            "score_records": attrs.get("rigor_records", []),
            "score_agent_name": "rigor",
        },
    )
    return {
        "rigor_records": persisted["score_records"],
        "rigor_batch_index": persisted["score_batch_index"],
    }


def persist_impact_batch(input_dict: dict[str, object], attrs: dict[str, object]) -> dict[str, object]:
    persisted = persist_score_batch(
        input_dict,
        {
            **attrs,
            "current_scoring_batch": attrs.get("impact_current_batch", []),
            "score_batch_index": attrs.get("impact_batch_index", 0),
            "score_records": attrs.get("impact_records", []),
            "score_agent_name": "impact",
        },
    )
    return {
        "impact_records": persisted["score_records"],
        "impact_batch_index": persisted["score_batch_index"],
    }


def _finalize_score_outputs(attrs: dict[str, object], agent_name: str, output_path: str) -> dict[str, object]:
    parsed_by_id = {
        str(item.get("paper_id") or ""): dict(item)
        for item in attrs.get("score_records", [])
        if isinstance(item, dict)
    }
    scores: list[dict[str, object]] = []
    for paper in attrs.get("paper_summaries", []):
        paper_id = str(paper.get("paper_id") or "")
        record = parsed_by_id.get(paper_id)
        if record is None:
            scores.append(_fallback_score_record(paper_id, agent_name, "Score record missing after batch loop."))
        else:
            scores.append(record)
    path = write_json(output_path, scores)
    return {
        f"{agent_name}_scores": scores,
        f"{agent_name}_scores_path": path,
    }


def finalize_novelty_scores(_: dict[str, object], attrs: dict[str, object]) -> dict[str, object]:
    return _finalize_score_outputs(
        {**attrs, "score_records": attrs.get("novelty_records", [])},
        "novelty",
        str(attrs["novelty_scores_path"]),
    )


def finalize_rigor_scores(_: dict[str, object], attrs: dict[str, object]) -> dict[str, object]:
    return _finalize_score_outputs(
        {**attrs, "score_records": attrs.get("rigor_records", [])},
        "rigor",
        str(attrs["rigor_scores_path"]),
    )


def finalize_impact_scores(_: dict[str, object], attrs: dict[str, object]) -> dict[str, object]:
    return _finalize_score_outputs(
        {**attrs, "score_records": attrs.get("impact_records", [])},
        "impact",
        str(attrs["impact_scores_path"]),
    )


def aggregate_and_prepare_brief(_: dict[str, object], attrs: dict[str, object]) -> dict[str, object]:
    aggregated = aggregate_scores({}, attrs)
    brief_attrs = {**attrs, **aggregated}
    prepared = prepare_brief_payload({}, brief_attrs)
    return {**aggregated, **prepared}


def aggregate_scores(_: dict[str, object], attrs: dict[str, object]) -> dict[str, object]:
    novelty = {item["paper_id"]: item for item in attrs.get("novelty_scores", [])}
    rigor = {item["paper_id"]: item for item in attrs.get("rigor_scores", [])}
    impact = {item["paper_id"]: item for item in attrs.get("impact_scores", [])}
    aggregated: list[dict[str, object]] = []

    for paper in attrs.get("paper_summaries", []):
        summary = paper.get("summary") or {}
        paper_id = str(paper.get("paper_id") or "")
        novelty_item = novelty.get(paper_id, {})
        rigor_item = rigor.get(paper_id, {})
        impact_item = impact.get(paper_id, {})

        novelty_score = parse_float(novelty_item.get("score"))
        rigor_score = parse_float(rigor_item.get("score"))
        impact_score = parse_float(impact_item.get("score"))
        weighted = novelty_score * 0.45 + rigor_score * 0.25 + impact_score * 0.30
        if novelty_item.get("is_incremental_ab"):
            weighted -= 0.75

        aggregated.append(
            {
                "paper_id": paper_id,
                "filename": paper.get("filename", ""),
                "title": summary.get("title") or paper.get("title_guess") or paper.get("filename"),
                "one_sentence_summary": summary.get("one_sentence_summary") or "",
                "paper_type": summary.get("paper_type") or "unknown",
                "keywords": summary.get("keywords") or [],
                "novelty_score": round(novelty_score, 2),
                "rigor_score": round(rigor_score, 2),
                "impact_score": round(impact_score, 2),
                "weighted_score": round(max(weighted, 0.0), 2),
                "incremental_ab_flag": bool(novelty_item.get("is_incremental_ab")),
                "novelty_reason": novelty_item.get("reasoning") or "",
                "rigor_reason": rigor_item.get("reasoning") or "",
                "impact_reason": impact_item.get("reasoning") or "",
                "strengths": sorted(
                    {
                        *(novelty_item.get("strengths") or []),
                        *(rigor_item.get("strengths") or []),
                        *(impact_item.get("strengths") or []),
                    }
                ),
                "concerns": sorted(
                    {
                        *(novelty_item.get("concerns") or []),
                        *(rigor_item.get("concerns") or []),
                        *(impact_item.get("concerns") or []),
                    }
                ),
            }
        )

    aggregated.sort(key=lambda item: item["weighted_score"], reverse=True)
    for rank, item in enumerate(aggregated, start=1):
        item["rank"] = rank
        if item["weighted_score"] >= 8.3:
            item["verdict"] = "top_highlight"
        elif item["weighted_score"] >= 7.2:
            item["verdict"] = "strong_watch"
        elif item["weighted_score"] >= 5.5:
            item["verdict"] = "incremental_or_niche"
        else:
            item["verdict"] = "low_signal"

    path = write_json(attrs["aggregated_scores_path"], aggregated)
    return {
        "aggregated_scores": aggregated,
        "aggregated_scores_path": path,
        "aggregated_scores_excerpt": excerpt(json.dumps(aggregated, ensure_ascii=False, indent=2), limit=12000),
    }


def prepare_brief_payload(_: dict[str, object], attrs: dict[str, object]) -> dict[str, object]:
    aggregated = list(attrs.get("aggregated_scores", []))
    inventory_by_id = {
        str(item.get("paper_id") or ""): item
        for item in attrs.get("paper_inventory", [])
        if isinstance(item, dict)
    }
    top_assets: list[dict[str, object]] = []
    for item in aggregated[: min(len(aggregated), 8)]:
        paper = inventory_by_id.get(str(item.get("paper_id") or ""), {})
        top_assets.append(
            {
                "paper_id": item.get("paper_id"),
                "title": item.get("title"),
                "pdf_path": paper.get("pdf_path", ""),
                "pdf_asset": paper.get("pdf_asset") or {},
                "preview_path": paper.get("preview_path", ""),
                "preview_image_asset": paper.get("preview_image_asset") or {},
                "tool_asset_root": attrs.get("tool_asset_root") or "",
            }
        )
    verdict_counts = Counter(item.get("verdict") or "unknown" for item in aggregated)
    paper_type_counts = Counter(item.get("paper_type") or "unknown" for item in aggregated)
    keyword_counts = Counter()
    for item in aggregated:
        keyword_counts.update(str(keyword).strip() for keyword in item.get("keywords") or [] if str(keyword).strip())

    payload = {
        "date_label": attrs.get("date_label"),
        "paper_count": attrs.get("paper_count", len(aggregated)),
        "stats": {
            "verdict_counts": dict(verdict_counts),
            "paper_type_counts": dict(paper_type_counts),
            "top_keywords": [{"keyword": key, "count": value} for key, value in keyword_counts.most_common(15)],
            "average_weighted_score": round(
                sum(parse_float(item.get("weighted_score")) for item in aggregated) / max(len(aggregated), 1),
                2,
            ),
        },
        "top_papers": aggregated[: min(len(aggregated), 8)],
        "top_paper_assets": top_assets,
        "watchlist": [
            {
                "paper_id": item["paper_id"],
                "title": item["title"],
                "weighted_score": item["weighted_score"],
                "verdict": item["verdict"],
                "keywords": item["keywords"],
                "one_sentence_summary": item["one_sentence_summary"],
            }
            for item in aggregated[8:20]
        ],
        "incremental_examples": [
            {
                "paper_id": item["paper_id"],
                "title": item["title"],
                "weighted_score": item["weighted_score"],
                "verdict": item["verdict"],
                "concerns": item["concerns"][:3],
            }
            for item in aggregated
            if item.get("incremental_ab_flag")
        ][:8],
    }
    return {
        "briefing_payload_json": json.dumps(payload, ensure_ascii=False, indent=2),
        "brief_pdf_assets": [item.get("pdf_asset") or {} for item in top_assets if item.get("pdf_asset")],
        "brief_preview_images": [
            item.get("preview_image_asset") or {}
            for item in top_assets
            if item.get("preview_image_asset")
        ],
    }


def persist_brief_outputs(_: dict[str, object], attrs: dict[str, object]) -> dict[str, object]:
    markdown = _strip_closing_tag(attrs.get("daily_brief_markdown") or "", "daily_brief_markdown")
    aggregated = list(attrs.get("aggregated_scores", []))
    brief_json = {
        "date_label": attrs.get("date_label"),
        "paper_count": attrs.get("paper_count", len(aggregated)),
        "top_papers": aggregated[: min(len(aggregated), 8)],
        "generated_markdown": markdown,
    }
    markdown_path = write_text(attrs["daily_brief_markdown_path"], markdown)
    json_path = write_json(attrs["daily_brief_json_path"], brief_json)
    return {
        "daily_brief_markdown": markdown,
        "daily_brief_markdown_path": markdown_path,
        "daily_brief_json": brief_json,
        "daily_brief_json_path": json_path,
    }


def persist_brief_and_finalize(_: dict[str, object], attrs: dict[str, object]) -> dict[str, object]:
    persisted = persist_brief_outputs({}, attrs)
    next_attrs = {**attrs, **persisted}
    finalized = finalize_daily_digest({}, next_attrs)
    return {**persisted, **finalized}


def finalize_daily_digest(_: dict[str, object], attrs: dict[str, object]) -> dict[str, object]:
    return {
        "run_dir": attrs.get("run_dir", ""),
        "paper_inventory_path": attrs.get("paper_inventory_path", ""),
        "paper_summaries_path": attrs.get("paper_summaries_path", ""),
        "aggregated_scores_path": attrs.get("aggregated_scores_path", ""),
        "daily_brief_json_path": attrs.get("daily_brief_json_path", ""),
        "daily_brief_markdown_path": attrs.get("daily_brief_markdown_path", ""),
    }
