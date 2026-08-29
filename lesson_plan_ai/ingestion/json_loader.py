import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_json_records(root: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for path in sorted(Path(root).rglob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, dict) and isinstance(value.get("questions"), list):
                values = value["questions"]
            else:
                values = value if isinstance(value, list) else [value]
            for index, item in enumerate(values):
                if not isinstance(item, dict):
                        raise TypeError(f"record {index} is not an object")
                records.append({**item, "source_filename": str(path)})
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            logger.warning("Could not parse %s: %s", path, exc)
            errors.append({"source_filename": str(path), "error": str(exc)})
    return records, errors
