from __future__ import annotations

import os
import traceback
from pathlib import Path
from typing import Any

try:
    from flask import Flask, abort, jsonify, request, send_file
except ImportError:  # pragma: no cover
    Flask = None  # type: ignore[assignment]
    abort = None  # type: ignore[assignment]
    jsonify = None  # type: ignore[assignment]
    request = None  # type: ignore[assignment]
    send_file = None  # type: ignore[assignment]

try:
    from flask_cors import CORS
except ImportError:  # pragma: no cover
    CORS = None  # type: ignore[assignment]

from .compiler import compile_document_to_graph, document_requires_model
from .ai_authoring import generate_authoring_field
from .key_pool import collect_document_key_pool, rename_document_key
from .schema import build_demo_document, parse_document, parse_document_with_errors, validate_document_errors
from .skill_packager import export_canvas_json, export_standard_skill, validate_standard_skill
from .validation import analyze_document


APP_ROOT = Path(__file__).resolve().parents[2]
EXPORT_ROOT = APP_ROOT / "exports"
FRONTEND_DIST = APP_ROOT / "frontend" / "dist"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 15081
DOWNLOADABLE_EXPORT_KEYS = {
    "skill_md_path",
    "skill_zip_path",
    "json_path",
}


def _exception_chain(err: Exception) -> list[dict[str, str]]:
    chain: list[dict[str, str]] = []
    current: Exception | None = err
    seen: set[int] = set()

    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(
            {
                "type": current.__class__.__name__,
                "message": str(current),
            }
        )
        next_exc = current.__cause__ or current.__context__
        current = next_exc if isinstance(next_exc, Exception) else None
    return chain


def _unwrap_retry_error(err: Exception) -> Exception | None:
    last_attempt = getattr(err, "last_attempt", None)
    if last_attempt is None or not hasattr(last_attempt, "exception"):
        return None
    try:
        unwrapped = last_attempt.exception()
    except Exception:
        return None
    return unwrapped if isinstance(unwrapped, Exception) else None


def _serialize_exception(err: Exception) -> dict[str, Any]:
    root = _unwrap_retry_error(err) or err
    chain = _exception_chain(root if root is not err else err)
    tb = "".join(traceback.format_exception(type(err), err, err.__traceback__))

    payload: dict[str, Any] = {
        "error": str(err),
        "error_type": err.__class__.__name__,
        "root_cause": str(root),
        "root_cause_type": root.__class__.__name__,
        "cause_chain": chain,
        "traceback": tb[-12000:],
    }

    if root is not err:
        payload["retry_error"] = {
            "message": str(err),
            "root_cause": str(root),
            "root_cause_type": root.__class__.__name__,
        }

    return payload


def _resolve_export_download_path(raw_path: str, file_key: str) -> Path:
    if file_key not in DOWNLOADABLE_EXPORT_KEYS:
        raise ValueError(f"Unsupported export file key: {file_key}")

    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = EXPORT_ROOT / candidate
    resolved = candidate.resolve()
    allowed_root = EXPORT_ROOT.resolve()

    if not (resolved == allowed_root or allowed_root in resolved.parents):
        raise ValueError("Requested export file is outside the export directory")
    if not resolved.is_file():
        raise FileNotFoundError(f"Export file not found: {resolved.name}")
    return resolved


def _serve_frontend_file(path: str = "index.html"):
    if send_file is None:
        raise RuntimeError("Flask send_file is not available.")
    if path.startswith("api/"):
        if abort is not None:
            abort(404)
        raise FileNotFoundError(path)

    requested = (FRONTEND_DIST / path).resolve()
    frontend_root = FRONTEND_DIST.resolve()
    index_path = FRONTEND_DIST / "index.html"

    if frontend_root in requested.parents and requested.is_file():
        return send_file(requested)
    if index_path.is_file():
        return send_file(index_path)
    raise FileNotFoundError("frontend build not found; run `npm run build` in applications/clawcanvas/frontend")


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def create_app() -> "Flask":
    if Flask is None:
        raise RuntimeError("Flask is not installed. Run `pip install -r requirements.txt` first.")

    app = Flask(__name__)
    if CORS is not None:
        CORS(app)

    @app.get("/api/health")
    def health():
        return jsonify({"ok": True, "service": "clawcanvas-backend"})

    @app.get("/api/demo")
    def demo():
        document = build_demo_document()
        return jsonify({"document": document.to_dict(), "key_pool": collect_document_key_pool(document)})

    @app.post("/api/validate")
    def validate_document_route():
        payload = request.get_json(force=True) or {}
        document, errors = parse_document_with_errors(payload.get("document") or payload)
        errors.extend(validate_document_errors(document))
        errors = list(dict.fromkeys(errors))
        analysis = analyze_document(document)
        return jsonify(
            {
                "ok": True,
                "valid": not errors,
                "document": document.to_dict(),
                "key_pool": analysis["key_pool"],
                "errors": errors,
                "warnings": analysis["warnings"],
                "summary": {
                    "node_count": len(document.nodes),
                    "edge_count": len(document.edges),
                    "error_count": len(errors),
                    "warning_count": len(analysis["warnings"]),
                },
            }
        )

    @app.post("/api/validate-skill")
    def validate_skill_route():
        payload = request.get_json(force=True) or {}
        document = parse_document(payload.get("document") or payload)
        result = export_standard_skill(
            document,
            export_root=EXPORT_ROOT,
            run_output=dict(payload.get("runOutput") or {}),
            warnings=list(payload.get("warnings") or []),
            runtime={},
        )
        validation = result.get("validation") or validate_standard_skill(
            result["skill_md_path"],
            skill_dir=result.get("skill_dir"),
        )
        return jsonify(
            {
                "ok": True,
                "valid": bool(validation.get("ok")),
                "validation": validation,
                "skill_md_path": result["skill_md_path"],
                "skill_zip_path": result["skill_zip_path"],
            }
        )

    @app.post("/api/key-pool/rename")
    def rename_key_route():
        payload = request.get_json(force=True) or {}
        document = parse_document(payload.get("document") or payload)
        old_key = str(payload.get("oldKey") or "")
        new_key = str(payload.get("newKey") or "")
        updated = rename_document_key(document, old_key, new_key)
        return jsonify(
            {
                "ok": True,
                "document": updated.to_dict(),
                "key_pool": collect_document_key_pool(updated),
            }
        )

    @app.post("/api/run")
    def run_document_route():
        payload = request.get_json(force=True) or {}
        document = parse_document(payload.get("document") or payload)
        runtime = payload.get("runtime") or {}
        api_key = str(runtime.get("apiKey") or "").strip()
        if document_requires_model(document) and not api_key:
            return jsonify({"ok": False, "error": "runtime.apiKey is required"}), 400

        model_name = str(runtime.get("modelName") or "gpt-4o-mini")
        base_url = str(runtime.get("baseUrl") or "").strip() or None
        graph, warnings = compile_document_to_graph(
            document,
            api_key=api_key,
            model_name=model_name,
            base_url=base_url,
        )
        output, attributes = graph.invoke(dict(document.inputs), attributes=dict(document.attributes))
        return jsonify(
            {
                "ok": True,
                "output": output,
                "attributes": attributes,
                "warnings": warnings.items,
                "runtime": {
                    "model_name": model_name,
                    "base_url": base_url,
                },
            }
        )

    @app.post("/api/export-skill")
    def export_skill_route():
        payload = request.get_json(force=True) or {}
        document = parse_document(payload.get("document") or payload)
        run_output = dict(payload.get("runOutput") or {})
        warnings = list(payload.get("warnings") or [])
        runtime = dict(payload.get("runtime") or {})

        result = export_standard_skill(
            document,
            export_root=EXPORT_ROOT,
            run_output=run_output,
            warnings=warnings,
            runtime=runtime,
        )
        return jsonify({"ok": True, **result})

    @app.post("/api/export-json")
    def export_json_route():
        payload = request.get_json(force=True) or {}
        document = parse_document(payload.get("document") or payload)
        run_output = dict(payload.get("runOutput") or {})
        warnings = list(payload.get("warnings") or [])

        result = export_canvas_json(
            document,
            export_root=EXPORT_ROOT,
            run_output=run_output,
            warnings=warnings,
        )
        return jsonify({"ok": True, **result})

    @app.post("/api/ai-authoring/field")
    def ai_authoring_field_route():
        payload = request.get_json(force=True) or {}
        document = parse_document(payload.get("document") or payload)
        result = generate_authoring_field(
            document,
            ai=dict(payload.get("ai") or {}),
            field=str(payload.get("field") or ""),
            mode=str(payload.get("mode") or "generate"),
            current_value=payload.get("currentValue") or "",
            locale=str(payload.get("locale") or "en"),
        )
        return jsonify({"ok": True, **result})

    @app.get("/api/download-export")
    def download_export_route():
        if send_file is None:
            raise RuntimeError("Flask send_file is not available.")

        raw_path = str(request.args.get("path") or "")
        file_key = str(request.args.get("key") or "")
        if not raw_path or not file_key:
            return jsonify({"ok": False, "error": "path and key are required"}), 400

        download_path = _resolve_export_download_path(raw_path, file_key)
        return send_file(download_path, as_attachment=True, download_name=download_path.name)

    @app.get("/")
    def serve_frontend_index():
        return _serve_frontend_file()

    @app.get("/<path:path>")
    def serve_frontend_asset_or_spa(path: str):
        return _serve_frontend_file(path)

    @app.errorhandler(Exception)
    def handle_exception(err: Exception):
        status = getattr(err, "code", None)
        if status is None:
            status = 400 if isinstance(err, (ValueError, KeyError)) else 500
        return jsonify({"ok": False, **_serialize_exception(err)}), status

    return app


def main() -> None:
    create_app().run(
        host=os.environ.get("CLAWCANVAS_HOST", DEFAULT_HOST),
        port=_env_int("CLAWCANVAS_PORT", _env_int("PORT", DEFAULT_PORT)),
        debug=_env_bool("CLAWCANVAS_DEBUG", False),
    )


if __name__ == "__main__":  # pragma: no cover
    main()
