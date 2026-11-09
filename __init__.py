"""`dag_monitor`: In-session slash command and telemetry viewer for DAG-ULW runs."""

from __future__ import annotations

import logging
from typing import Any, Optional

try:
    from . import render, telemetry
except ImportError:
    import render  # type: ignore
    import telemetry  # type: ignore

logger = logging.getLogger("hermes_cli.plugins.dag_monitor")


def _slash_handler(raw_args: str = "", *args: Any, **kwargs: Any) -> str:
    """Handle /dag-monitor [run_id] slash command execution."""
    arg = raw_args.strip() if isinstance(raw_args, str) else ""
    run_id = arg if arg else None

    data = telemetry.get_dag_telemetry(run_id=run_id)
    html_path = render.save_dashboard_html(data)

    summary = data.get("summary", {})
    tokens = summary.get("tokens", {})
    savings = summary.get("savings", {})
    run_id_val = data.get("run_id") or "unknown"
    status_val = data.get("status") or "unknown"
    completed_nodes = summary.get("completed_nodes", 0)
    total_nodes = summary.get("total_nodes", 0)

    inp_str = render._fmt_tokens(tokens.get("input_tokens", 0))
    out_str = render._fmt_tokens(tokens.get("output_tokens", 0))
    cache_str = render._fmt_tokens(tokens.get("cache_read_tokens", 0))
    tot_str = render._fmt_tokens(tokens.get("total_tokens", 0))
    cache_den = (tokens.get("input_tokens", 0) or 0) + (tokens.get("cache_read_tokens", 0) or 0)
    cache_pct = f"{(tokens.get('cache_read_tokens', 0) / cache_den * 100):.1f}%" if cache_den > 0 else "0.0%"

    lines = [
        f'::preview{{file="{html_path}"}}',
        "",
        f"### DAG Telemetry: `{run_id_val}` (Status: {status_val})",
        f"- **Nodes**: {completed_nodes}/{total_nodes} completed",
        f"- **Tokens**: {tot_str} total ({inp_str} in, {out_str} out, {cache_str} cache read — {cache_pct} hit rate)",
    ]
    if tokens.get("api_call_count"):
        lines.append(f"- **API Calls**: {tokens.get('api_call_count')}")

    if savings:
        lines.append("- **Relative Quota Savings (vs Claude Opus 4.6)**:")
        for model_key, s in savings.items():
            label = s.get("compared_model", model_key)
            saved_usd = render._fmt_usd(s.get("saved_usd", 0.0), precision=2)
            pct = s.get("saved_pct", 0.0)
            mult = s.get("multiplier", 1.0)
            lines.append(f"  * **{label}**: saved {saved_usd} ({pct:.1f}% savings, {mult:.1f}x cheaper)")

    return "\n".join(lines)


def register(ctx: Any) -> None:
    """Register the dag-monitor plugin into Hermes."""
    ctx.register_command(
        "dag-monitor",
        handler=_slash_handler,
        description="Visualiza telemetria e economia de cotas do DAG-ULW.",
    )
