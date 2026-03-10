"""
Generate a self-contained HTML report from a run's results directory.
Inlines all PNGs as base64 so the file can be shared without the images folder.
"""

import base64
import json
import os
import logging
from typing import Any, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

REPORT_IMAGES = [
    "all_models_trellis.png",
    "top_3_models_comparison.png",
    "model_performance.png",
    "model_radar.png",
    "top_3_residuals.png",
    "feature_importance.png",
]


def _img_to_base64(path: str) -> Optional[str]:
    """Read a PNG and return base64 data URL string, or None if file missing."""
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        return f"data:image/png;base64,{b64}"
    except (OSError, ValueError) as e:
        logger.warning("Could not inline image %s: %s", path, e)
        return None


def _results_table_html(results_df: pd.DataFrame) -> str:
    """Render results DataFrame as HTML table (model, composite_score, rmse, mae, r2, mase, mape)."""
    cols = [c for c in ["model", "composite_score", "rmse", "mae", "r2", "mase", "mape"] if c in results_df.columns]
    if not cols:
        return "<p>No metrics table.</p>"
    df = results_df[cols].head(20)
    df = df.round(4)
    return df.to_html(index=False, classes="results-table", border=0)


def _tuning_summary_html(results_summary: Dict[str, Any]) -> str:
    """Optional section for tuned best params."""
    if not results_summary.get("tuned_best_params"):
        return ""
    lines = ["<h3>Tuning summary</h3>", "<p>Best parameters (validation-selected):</p>", "<ul>"]
    for key, params in results_summary["tuned_best_params"].items():
        if isinstance(params, dict):
            lines.append(f"<li><strong>{key}</strong>: {json.dumps(params)}</li>")
        else:
            lines.append(f"<li><strong>{key}</strong>: {params}</li>")
    lines.append("</ul>")
    if results_summary.get("parameter_selection"):
        note = results_summary["parameter_selection"].get("note", "")
        if note:
            lines.append(f"<p><em>{note}</em></p>")
    return "\n".join(lines)


def generate_html_report(
    results_dir: str,
    results_summary: Dict[str, Any],
    results_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> str:
    """
    Generate a self-contained report.html in results_dir.
    Inlines all PNGs as base64. Includes dataset info, judgment, metrics table, and tuning summary.
    """
    if save_path is None:
        save_path = os.path.join(results_dir, "report.html")

    # Inline images
    images_html = []
    for name in REPORT_IMAGES:
        path = os.path.join(results_dir, name)
        data = _img_to_base64(path)
        if data:
            images_html.append(f'<h3>{name}</h3><img src="{data}" alt="{name}" style="max-width:100%;"/>')
        else:
            images_html.append(f"<h3>{name}</h3><p>(image not found)</p>")

    dataset = results_summary.get("dataset", {})
    judgment = results_summary.get("best_judgment", "")
    table_html = _results_table_html(results_df)
    tuning_html = _tuning_summary_html(results_summary)

    top3 = results_df.head(3)
    top3_lines = []
    for i, (_, row) in enumerate(top3.iterrows(), 1):
        top3_lines.append(
            f"<li>{i}. {row['model']} — composite: {row.get('composite_score', '')}, "
            f"RMSE: {row.get('rmse', '')}, R²: {row.get('r2', '')}</li>"
        )
    top3_html = "<ul>" + "".join(top3_lines) + "</ul>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Time Series Forecast Report — {dataset.get('name', 'Report')}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; max-width: 1200px; }}
    h1 {{ border-bottom: 1px solid #ccc; }}
    .results-table {{ border-collapse: collapse; }}
    .results-table th, .results-table td {{ padding: 6px 12px; text-align: left; border: 1px solid #ddd; }}
    .results-table th {{ background: #f5f5f5; }}
    .judgment {{ background: #f9f9f9; padding: 1rem; border-left: 4px solid #1f77b4; margin: 1rem 0; }}
    img {{ margin: 1rem 0; }}
  </style>
</head>
<body>
  <h1>Time Series Forecast Report</h1>
  <p><strong>Dataset:</strong> {dataset.get('name', '')} — {dataset.get('total_samples', '')} samples
     ({dataset.get('training_samples', '')} train / {dataset.get('testing_samples', '')} test)</p>

  <div class="judgment">
    <h2>Recommendation</h2>
    <p>{judgment}</p>
  </div>

  <h2>Top 3 models</h2>
  {top3_html}

  <h2>Metrics (all models)</h2>
  {table_html}

  {tuning_html}

  <h2>Charts</h2>
  {"".join(images_html)}
</body>
</html>
"""

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info("HTML report saved to '%s'", save_path)
    return save_path
