"""HTML report export helpers."""

from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd

from ..constants import DISCLAIMER


def export_html_report(title: str, sections: dict[str, str | pd.DataFrame], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    body = [f"<h1>{escape(title)}</h1>", f"<p><strong>{escape(DISCLAIMER)}</strong></p>"]
    for heading, content in sections.items():
        body.append(f"<h2>{escape(heading)}</h2>")
        if isinstance(content, pd.DataFrame):
            body.append(content.to_html(index=False, escape=True))
        else:
            body.append(f"<p>{escape(str(content))}</p>")
    html = "<!doctype html><html><head><meta charset='utf-8'><title>{}</title></head><body>{}</body></html>".format(
        escape(title),
        "\n".join(body),
    )
    output_path.write_text(html, encoding="utf-8")
    return output_path

