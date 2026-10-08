"""Write reports and guidance to disk (shared by `llb report`, `llb guide` and `llb run`)."""

from collections.abc import Iterable
from pathlib import Path

from llmbench.report.data import ReportData
from llmbench.report.markdown import render_guidance, render_markdown


def write_reports(d: ReportData, out_dir: Path, formats: Iterable[str]) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for fmt in formats:
        if fmt == "md":
            path = out_dir / "report.md"
            path.write_text(render_markdown(d), encoding="utf-8")
            written.append(path)
        elif fmt == "html":
            from llmbench.report.html import render_html  # noqa: PLC0415 (optional extra)

            path = out_dir / "report.html"
            path.write_text(render_html(d), encoding="utf-8")
            written.append(path)
        elif fmt == "png":
            from llmbench.report.html import render_png  # noqa: PLC0415 (optional extra)

            written += render_png(d, out_dir / "charts")
        else:
            raise ValueError(f"unknown report format {fmt!r} (html, md, png)")
    return written


def write_guidance(d: ReportData, out_dir: Path) -> Path:
    path = out_dir / "guidance.md"
    path.write_text(render_guidance(d), encoding="utf-8")
    return path
