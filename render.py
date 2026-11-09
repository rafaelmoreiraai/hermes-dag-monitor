"""`dag_monitor.render`: Responsive HTML dashboard renderer for DAG node token consumption.

Generates a standalone, dark-themed, ultra-clean HTML dashboard designed for
Hermes Desktop `::preview`, displaying real-time DAG execution topology, per-node
token metrics, cache hits, multi-model cost benchmarks, and OpenCodex gateway telemetry.
"""

from __future__ import annotations

import html
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


DEFAULT_SCRATCH_DIR = Path("/root/.hermes/cache/scratch")


def _fmt_tokens(n: int | float | None) -> str:
    """Format token count into compact human readable string (e.g. 1.25M, 45.2k)."""
    if n is None:
        return "0"
    n_val = int(n)
    if n_val >= 1_000_000:
        return f"{n_val / 1_000_000:.2f}M"
    if n_val >= 1_000:
        return f"{n_val / 1_000:.1f}k"
    return f"{n_val:,}"


def _fmt_usd(val: float | None, precision: int = 4) -> str:
    """Format USD cost."""
    if val is None:
        return "$0.00"
    return f"${val:,.{precision}f}"


def _fmt_duration(seconds: float | None) -> str:
    """Format duration in seconds into human-readable string."""
    if seconds is None or seconds <= 0:
        return "0s"
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m}m {s:02d}s"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}h {m:02d}m {s:02d}s"


def _fmt_timestamp(ts: float | None) -> str:
    """Format unix timestamp into ISO / readable time."""
    if not ts:
        return "N/A"
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(ts))
    except Exception:
        return str(ts)


def _category_style(category: str | None) -> tuple[str, str, str]:
    """Return (text_color, bg_color, border_color) for DAG node category."""
    cat = (category or "").lower()
    if "quick" in cat:
        return ("#38bdf8", "rgba(56, 189, 248, 0.12)", "rgba(56, 189, 248, 0.35)")
    if "deep" in cat:
        return ("#c084fc", "rgba(192, 132, 252, 0.12)", "rgba(192, 132, 252, 0.35)")
    if "ultra" in cat:
        return ("#f472b6", "rgba(244, 114, 182, 0.12)", "rgba(244, 114, 182, 0.35)")
    if "visual" in cat or "frontend" in cat:
        return ("#34d399", "rgba(52, 211, 153, 0.12)", "rgba(52, 211, 153, 0.35)")
    if "review" in cat or "verify" in cat:
        return ("#fbbf24", "rgba(251, 191, 36, 0.12)", "rgba(251, 191, 36, 0.35)")
    return ("#94a3b8", "rgba(148, 163, 184, 0.12)", "rgba(148, 163, 184, 0.35)")


def _status_badge_info(status: str | None) -> tuple[str, str, str, str]:
    """Return (label, text_color, bg_color, dot_class) for status."""
    st = (status or "pending").lower()
    if st in ("completed", "accepted", "done", "success"):
        return ("Completed", "#34d399", "rgba(52, 211, 153, 0.14)", "dot-completed")
    if st in ("running", "active", "in_progress"):
        return ("Running", "#38bdf8", "rgba(56, 189, 248, 0.14)", "dot-running")
    if st in ("ready", "queued"):
        return ("Ready", "#fbbf24", "rgba(251, 191, 36, 0.14)", "dot-ready")
    if st in ("failed", "error", "rejected"):
        return ("Failed", "#f87171", "rgba(248, 113, 113, 0.14)", "dot-failed")
    return ("Pending", "#94a3b8", "rgba(148, 163, 184, 0.1)", "dot-pending")


def render_dag_dashboard(telemetry_data: dict[str, Any]) -> str:
    """Generate a standalone, dark-themed, ultra-clean HTML dashboard from DAG telemetry.

    Parameters:
        telemetry_data: Dictionary returned by `telemetry.get_dag_telemetry()`.

    Returns:
        Complete HTML string with embedded responsive CSS, SVG icons, and client-side interactions.
    """
    if not isinstance(telemetry_data, dict):
        telemetry_data = {}

    run_id = str(telemetry_data.get("run_id") or "dag-run")
    name = str(telemetry_data.get("name") or run_id)
    goal = str(telemetry_data.get("goal") or "")
    session_id = str(telemetry_data.get("session_id") or "")
    raw_status = str(telemetry_data.get("status") or "running")
    defined_at = telemetry_data.get("defined_at")
    updated_at = telemetry_data.get("updated_at")

    # Compute wall duration
    if defined_at and updated_at and updated_at >= defined_at:
        wall_duration_s = float(updated_at - defined_at)
    else:
        wall_duration_s = 0.0

    summary = telemetry_data.get("summary") or {}
    total_nodes = int(summary.get("total_nodes") or 0)
    completed_nodes = int(summary.get("completed_nodes") or 0)
    running_nodes = int(summary.get("running_nodes") or 0)
    ready_nodes = int(summary.get("ready_nodes") or 0)
    pending_nodes = int(summary.get("pending_nodes") or 0)
    failed_nodes = int(summary.get("failed_nodes") or 0)

    tokens = summary.get("tokens") or {}
    inp_tok = int(tokens.get("input_tokens") or 0)
    out_tok = int(tokens.get("output_tokens") or 0)
    cache_tok = int(tokens.get("cache_read_tokens") or 0)
    reason_tok = int(tokens.get("reasoning_tokens") or 0)
    total_tok = int(tokens.get("total_tokens") or 0)
    api_calls = int(tokens.get("api_call_count") or 0)

    # Cache hit rate %: cache / (input + cache)
    cache_denominator = inp_tok + cache_tok
    if cache_denominator > 0:
        cache_hit_pct = round((cache_tok / cache_denominator) * 100.0, 1)
    else:
        cache_hit_pct = 0.0

    # Costs and Savings
    costs_usd = summary.get("costs_usd") or {}
    savings = summary.get("savings") or {}

    opus_cost = float(costs_usd.get("claude-opus-4-6") or 0.0)
    sonnet_cost = float(costs_usd.get("claude-sonnet-4-6") or 0.0)
    gemini_cost = float(costs_usd.get("gemini-3.8-flash") or 0.0)
    deepseek_cost = float(costs_usd.get("deepseek-flash") or 0.0)
    gpt_cost = float(costs_usd.get("gpt-5.6-sol") or 0.0)

    # Primary runtime cost (actual execution)
    gemini_savings = savings.get("gemini-3.8-flash") or {}
    net_saved_usd = float(gemini_savings.get("saved_usd") or max(0.0, opus_cost - gemini_cost))
    net_saved_pct = float(gemini_savings.get("saved_pct") or ((net_saved_usd / opus_cost * 100.0) if opus_cost > 0 else 0.0))
    cost_multiplier = float(gemini_savings.get("multiplier") or ((opus_cost / gemini_cost) if gemini_cost > 0 else 1.0))

    # OpenCodex Gateway summary
    oc_telemetry = telemetry_data.get("opencodex_telemetry") or {}
    oc_avail = bool(oc_telemetry.get("available"))
    oc_records = int(oc_telemetry.get("records_count") or 0)
    oc_providers = oc_telemetry.get("providers") or []
    oc_models = oc_telemetry.get("models") or []
    oc_duration_ms = int(oc_telemetry.get("total_duration_ms") or 0)
    oc_tokens = oc_telemetry.get("tokens") or {}
    oc_total_tokens = int(oc_tokens.get("total_tokens") or 0)

    # Topology
    topology = telemetry_data.get("topology") or {}
    critical_path = topology.get("critical_path") or []
    bottlenecks = {b.get("node"): b.get("blocks") for b in topology.get("bottlenecks") or [] if isinstance(b, dict)}
    layers = topology.get("layers") or []

    # Nodes dict
    nodes_dict = telemetry_data.get("nodes") or {}

    # Cost comparison bars data setup
    comparison_items = [
        ("Claude Opus 4.6 (Frontier Baseline)", opus_cost, "#a855f7", "#7c3aed", "Frontier Ref"),
        ("GPT-5.6 Sol (Anthropic Alt)", gpt_cost, "#ec4899", "#db2777", "Frontier"),
        ("Claude Sonnet 4.6 (Mid-Tier)", sonnet_cost, "#3b82f6", "#2563eb", "Standard"),
        ("Gemini 3.8 Flash (Execution Runtime)", gemini_cost, "#10b981", "#059669", "Runtime"),
        ("DeepSeek Flash (Ultra-Economy)", deepseek_cost, "#06b6d4", "#0891b2", "Economy"),
    ]
    max_benchmark_cost = max([c[1] for c in comparison_items] + [0.01])

    # Status badge styling for main header
    st_label, st_color, st_bg, st_dot = _status_badge_info(raw_status)

    # Render Node cards HTML
    node_cards_html = []
    for node_id, node in nodes_dict.items():
        n_label = str(node.get("label") or node_id)
        n_category = str(node.get("category") or "general")
        n_role = str(node.get("role") or "task")
        n_status = str(node.get("status") or "pending")
        n_model = str(node.get("model") or "agy-gemini-3.8-flash")
        n_provider = str(node.get("provider") or "custom")
        n_duration_s = float(node.get("duration_s") or 0.0)
        n_attempts = int(node.get("attempts_count") or 1)
        n_depends = node.get("depends_on") or []
        n_writes = node.get("write_scope") or []

        n_tok = node.get("tokens") or {}
        n_in = int(n_tok.get("input_tokens") or 0)
        n_out = int(n_tok.get("output_tokens") or 0)
        n_cache = int(n_tok.get("cache_read_tokens") or 0)
        n_reason = int(n_tok.get("reasoning_tokens") or 0)
        n_tot = int(n_tok.get("total_tokens") or 0)
        n_calls = int(n_tok.get("api_call_count") or 0)

        n_costs = node.get("costs_usd") or {}
        n_gemini_cost = float(n_costs.get("gemini-3.8-flash") or 0.0)
        n_opus_cost = float(n_costs.get("claude-opus-4-6") or 0.0)
        n_savings = node.get("savings_vs_opus") or {}
        n_saved_usd = float(n_savings.get("saved_usd") or max(0.0, n_opus_cost - n_gemini_cost))
        n_saved_pct = float(n_savings.get("saved_pct") or ((n_saved_usd / n_opus_cost * 100.0) if n_opus_cost > 0 else 0.0))

        # Node status styles
        nst_label, nst_color, nst_bg, nst_dot = _status_badge_info(n_status)
        cat_color, cat_bg, cat_border = _category_style(n_category)

        # Critical path badge
        is_crit = node_id in critical_path
        blocks_count = bottlenecks.get(node_id, 0)

        # Node cache hit %
        n_cache_denom = n_in + n_cache
        n_cache_pct = round((n_cache / n_cache_denom * 100.0), 1) if n_cache_denom > 0 else 0.0

        deps_badges = "".join(f'<span class="dep-pill">{html.escape(d)}</span>' for d in n_depends) or '<span class="text-muted">None</span>'
        writes_list = "".join(f'<div class="write-file-pill" title="{html.escape(w)}">{html.escape(Path(w).name)}</div>' for w in n_writes) or '<span class="text-muted">None</span>'

        card_html = f"""
        <div class="node-card" data-node-id="{html.escape(node_id)}" data-status="{html.escape(n_status.lower())}" data-category="{html.escape(n_category.lower())}">
          <div class="node-header">
            <div class="node-title-group">
              <div class="node-id-row">
                <span class="node-id">{html.escape(node_id)}</span>
                {f'<span class="badge-critical" title="Node is on the DAG critical path">Critical Path</span>' if is_crit else ''}
                {f'<span class="badge-bottleneck" title="Directly blocks downstream nodes">Blocks {blocks_count}</span>' if blocks_count else ''}
              </div>
              <div class="node-label">{html.escape(n_label)}</div>
            </div>
            <div class="node-badges">
              <span class="badge-category" style="color:{cat_color}; background:{cat_bg}; border:1px solid {cat_border};">{html.escape(n_category)}</span>
              <span class="badge-role">{html.escape(n_role)}</span>
              <span class="badge-status" style="color:{nst_color}; background:{nst_bg};">
                <span class="status-dot {nst_dot}"></span>
                {nst_label}
              </span>
            </div>
          </div>

          <div class="node-meta-grid">
            <div class="node-meta-item">
              <span class="meta-label">Model Runtime</span>
              <span class="meta-value font-mono" title="{html.escape(n_model)} ({html.escape(n_provider)})">{html.escape(n_model)}</span>
            </div>
            <div class="node-meta-item">
              <span class="meta-label">Duration</span>
              <span class="meta-value font-mono">{_fmt_duration(n_duration_s)}</span>
            </div>
            <div class="node-meta-item">
              <span class="meta-label">Calls / Attempts</span>
              <span class="meta-value font-mono">{n_calls} reqs / {n_attempts} att</span>
            </div>
            <div class="node-meta-item">
              <span class="meta-label">Cache Hit</span>
              <span class="meta-value font-mono" style="color:#818cf8;">{n_cache_pct}%</span>
            </div>
          </div>

          <div class="token-strip">
            <div class="token-col">
              <span class="token-lbl">Total Tokens</span>
              <span class="token-val font-mono">{_fmt_tokens(n_tot)}</span>
              <span class="token-sub font-mono">{n_tot:,}</span>
            </div>
            <div class="token-col">
              <span class="token-lbl">Prompt In</span>
              <span class="token-val font-mono text-cyan">{_fmt_tokens(n_in)}</span>
              <span class="token-sub font-mono">{n_in:,}</span>
            </div>
            <div class="token-col">
              <span class="token-lbl">Cache Read</span>
              <span class="token-val font-mono text-indigo">{_fmt_tokens(n_cache)}</span>
              <span class="token-sub font-mono">{n_cache:,}</span>
            </div>
            <div class="token-col">
              <span class="token-lbl">Completion Out</span>
              <span class="token-val font-mono text-emerald">{_fmt_tokens(n_out)}</span>
              <span class="token-sub font-mono">{n_out:,}</span>
            </div>
            {f'''<div class="token-col">
              <span class="token-lbl">Reasoning</span>
              <span class="token-val font-mono text-pink">{_fmt_tokens(n_reason)}</span>
              <span class="token-sub font-mono">{n_reason:,}</span>
            </div>''' if n_reason > 0 else ''}
          </div>

          <div class="node-cost-card">
            <div class="cost-row">
              <div class="cost-item">
                <span class="cost-sublabel">Execution Cost</span>
                <span class="cost-figure text-emerald font-mono">{_fmt_usd(n_gemini_cost, 4)}</span>
              </div>
              <div class="cost-vs">vs</div>
              <div class="cost-item">
                <span class="cost-sublabel">Opus 4.6 Baseline</span>
                <span class="cost-figure text-muted font-mono">{_fmt_usd(n_opus_cost, 4)}</span>
              </div>
              <div class="cost-savings-pill" title="Estimated savings compared to frontier baseline">
                <span class="savings-val">+{_fmt_usd(n_saved_usd, 3)}</span>
                <span class="savings-pct font-mono">({n_saved_pct:.1f}% saved)</span>
              </div>
            </div>
          </div>

          <div class="node-footer">
            <div class="node-footer-section">
              <span class="footer-sec-label">Dependencies:</span>
              <div class="deps-container">{deps_badges}</div>
            </div>
            <div class="node-footer-section">
              <span class="footer-sec-label">Write Scope:</span>
              <div class="writes-container">{writes_list}</div>
            </div>
          </div>
        </div>
        """
        node_cards_html.append(card_html)

    # Render layers flow
    layers_html_items = []
    for idx, layer_nodes in enumerate(layers):
        pills = "".join(
            f'<div class="layer-node-pill {"is-crit" if nid in critical_path else ""}">'
            f'<span class="layer-node-id font-mono">{html.escape(nid)}</span>'
            f'<span class="layer-node-st">{html.escape(nodes_dict.get(nid, {}).get("status", "pending"))}</span>'
            f'</div>'
            for nid in layer_nodes
        )
        layers_html_items.append(
            f'<div class="dag-layer-box">'
            f'<div class="layer-title">Layer {idx}</div>'
            f'<div class="layer-nodes-flow">{pills}</div>'
            f'</div>'
        )
        if idx < len(layers) - 1:
            layers_html_items.append('<div class="dag-layer-arrow">→</div>')
    layers_flow_html = "".join(layers_html_items)

    # Cost comparison bars HTML
    comparison_bars_html = []
    for label, cost_val, grad_start, grad_end, role_tag in comparison_items:
        pct_width = max(3.0, min(100.0, (cost_val / max_benchmark_cost) * 100.0))
        saved_vs_opus = max(0.0, opus_cost - cost_val)
        saved_pct_str = f"(-{(saved_vs_opus / opus_cost * 100):.1f}%)" if (opus_cost > 0 and cost_val < opus_cost) else ""
        comparison_bars_html.append(f"""
        <div class="benchmark-row">
          <div class="benchmark-info">
            <div class="benchmark-title-wrap">
              <span class="benchmark-name">{html.escape(label)}</span>
              <span class="benchmark-tag">{html.escape(role_tag)}</span>
            </div>
            <div class="benchmark-cost-wrap">
              <span class="benchmark-cost font-mono">{_fmt_usd(cost_val, 4)}</span>
              <span class="benchmark-diff font-mono">{saved_pct_str}</span>
            </div>
          </div>
          <div class="bar-track">
            <div class="bar-fill" style="width: {pct_width:.1f}%; background: linear-gradient(90deg, {grad_start}, {grad_end});"></div>
          </div>
        </div>
        """)

    # Full HTML template
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>DAG Token Monitor - {html.escape(run_id)}</title>
  <style>
    :root {{
      --bg-canvas: #080b11;
      --bg-surface: #0e1524;
      --bg-surface-elevated: #141f36;
      --bg-card: rgba(18, 26, 44, 0.75);
      --border-subtle: rgba(255, 255, 255, 0.08);
      --border-hover: rgba(56, 189, 248, 0.35);
      --text-main: #f1f5f9;
      --text-muted: #94a3b8;
      --text-dim: #64748b;
      --accent-cyan: #38bdf8;
      --accent-emerald: #10b981;
      --accent-indigo: #818cf8;
      --accent-purple: #c084fc;
      --accent-pink: #f472b6;
      --accent-amber: #fbbf24;
      --accent-rose: #f87171;
      --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Inter, sans-serif;
      --font-mono: ui-monospace, "SF Mono", "Cascadia Code", "Source Code Pro", Menlo, Monaco, Consolas, monospace;
    }}

    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}

    body {{
      background: radial-gradient(circle at 10% 20%, #0d1527 0%, var(--bg-canvas) 80%);
      color: var(--text-main);
      font-family: var(--font-sans);
      font-size: 14px;
      line-height: 1.5;
      padding: 24px;
      min-height: 100vh;
      -webkit-font-smoothing: antialiased;
    }}

    .container {{
      max-width: 1360px;
      margin: 0 auto;
      display: flex;
      flex-direction: column;
      gap: 20px;
    }}

    /* Header */
    .dashboard-header {{
      background: var(--bg-card);
      backdrop-filter: blur(12px);
      border: 1px solid var(--border-subtle);
      border-radius: 14px;
      padding: 20px 24px;
      display: flex;
      flex-direction: column;
      gap: 16px;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.36);
    }}

    .header-top {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 16px;
    }}

    .brand-group {{
      display: flex;
      align-items: center;
      gap: 10px;
    }}

    .plugin-pill {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      padding: 4px 10px;
      border-radius: 6px;
      background: rgba(56, 189, 248, 0.12);
      color: var(--accent-cyan);
      border: 1px solid rgba(56, 189, 248, 0.25);
    }}

    .preview-pill {{
      font-size: 11px;
      font-weight: 600;
      padding: 4px 8px;
      border-radius: 6px;
      background: rgba(255, 255, 255, 0.06);
      color: var(--text-muted);
      border: 1px solid var(--border-subtle);
    }}

    .title-area {{
      display: flex;
      flex-direction: column;
      gap: 4px;
    }}

    .title-area h1 {{
      font-size: 22px;
      font-weight: 700;
      color: #ffffff;
      letter-spacing: -0.02em;
    }}

    .title-area .goal-desc {{
      color: var(--text-muted);
      font-size: 13px;
      max-width: 820px;
    }}

    .header-badges {{
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }}

    .status-badge-lg {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 6px 14px;
      border-radius: 20px;
      font-weight: 600;
      font-size: 13px;
      border: 1px solid var(--border-subtle);
    }}

    .status-dot {{
      width: 8px;
      height: 8px;
      border-radius: 50%;
      display: inline-block;
    }}

    .dot-running {{
      background: var(--accent-cyan);
      box-shadow: 0 0 10px var(--accent-cyan);
      animation: pulse 1.6s infinite ease-in-out;
    }}

    .dot-completed {{
      background: var(--accent-emerald);
      box-shadow: 0 0 8px var(--accent-emerald);
    }}

    .dot-ready {{
      background: var(--accent-amber);
    }}

    .dot-pending {{
      background: var(--text-dim);
    }}

    .dot-failed {{
      background: var(--accent-rose);
      box-shadow: 0 0 10px var(--accent-rose);
    }}

    @keyframes pulse {{
      0%, 100% {{ transform: scale(1); opacity: 0.9; }}
      50% {{ transform: scale(1.35); opacity: 0.4; }}
    }}

    .header-meta-bar {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 12px;
      padding-top: 14px;
      border-top: 1px solid var(--border-subtle);
      font-size: 12px;
      color: var(--text-muted);
    }}

    .meta-pills-list {{
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }}

    .pill-info {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 10px;
      border-radius: 6px;
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
    }}

    .pill-info strong {{
      color: var(--text-main);
    }}

    /* OpenCodex Gateway summary card */
    .gateway-summary-bar {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 12px;
      padding: 10px 16px;
      border-radius: 10px;
      background: linear-gradient(90deg, rgba(56, 189, 248, 0.08) 0%, rgba(129, 140, 248, 0.06) 100%);
      border: 1px solid rgba(56, 189, 248, 0.2);
      font-size: 12px;
    }}

    .gateway-left {{
      display: flex;
      align-items: center;
      gap: 10px;
    }}

    .gateway-pulse-pill {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-weight: 600;
      color: var(--accent-cyan);
    }}

    .gateway-details {{
      display: flex;
      align-items: center;
      gap: 14px;
      color: var(--text-muted);
      flex-wrap: wrap;
    }}

    /* Summary KPI Cards */
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 16px;
    }}

    .kpi-card {{
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: 12px;
      padding: 18px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      gap: 10px;
      position: relative;
      overflow: hidden;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
      transition: transform 0.2s, border-color 0.2s;
    }}

    .kpi-card:hover {{
      border-color: var(--border-hover);
      transform: translateY(-2px);
    }}

    .kpi-card.highlight-savings {{
      border: 1px solid rgba(16, 185, 129, 0.4);
      background: radial-gradient(circle at top right, rgba(16, 185, 129, 0.12) 0%, var(--bg-card) 75%);
    }}

    .kpi-title {{
      font-size: 12px;
      font-weight: 600;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }}

    .kpi-val {{
      font-size: 28px;
      font-weight: 700;
      color: #ffffff;
      line-height: 1.1;
      letter-spacing: -0.02em;
    }}

    .kpi-val.text-emerald {{
      color: var(--accent-emerald);
    }}

    .kpi-subtext {{
      font-size: 12px;
      color: var(--text-muted);
      display: flex;
      flex-direction: column;
      gap: 4px;
    }}

    .kpi-breakdown-row {{
      display: flex;
      justify-content: space-between;
      padding-top: 4px;
      border-top: 1px solid rgba(255, 255, 255, 0.05);
      font-size: 11px;
    }}

    .progress-bar-wrap {{
      width: 100%;
      height: 6px;
      background: rgba(255, 255, 255, 0.08);
      border-radius: 999px;
      overflow: hidden;
      margin-top: 6px;
    }}

    .progress-bar-fill {{
      height: 100%;
      background: linear-gradient(90deg, var(--accent-indigo), var(--accent-cyan));
      border-radius: 999px;
    }}

    /* Comparison Bar Section */
    .comparison-section {{
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: 14px;
      padding: 20px 24px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }}

    .section-header-row {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 10px;
    }}

    .section-title {{
      font-size: 15px;
      font-weight: 700;
      color: #ffffff;
      display: flex;
      align-items: center;
      gap: 8px;
    }}

    .section-badge {{
      font-size: 11px;
      font-weight: 600;
      padding: 2px 8px;
      border-radius: 12px;
      background: rgba(16, 185, 129, 0.15);
      color: var(--accent-emerald);
      border: 1px solid rgba(16, 185, 129, 0.3);
    }}

    .benchmark-list {{
      display: flex;
      flex-direction: column;
      gap: 12px;
    }}

    .benchmark-row {{
      display: flex;
      flex-direction: column;
      gap: 6px;
    }}

    .benchmark-info {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 12px;
    }}

    .benchmark-title-wrap {{
      display: flex;
      align-items: center;
      gap: 8px;
    }}

    .benchmark-name {{
      font-weight: 600;
      color: var(--text-main);
    }}

    .benchmark-tag {{
      font-size: 10px;
      padding: 1px 6px;
      border-radius: 4px;
      background: rgba(255, 255, 255, 0.08);
      color: var(--text-dim);
      font-family: var(--font-mono);
    }}

    .benchmark-cost-wrap {{
      display: flex;
      align-items: center;
      gap: 8px;
    }}

    .benchmark-cost {{
      font-weight: 700;
      color: #ffffff;
    }}

    .benchmark-diff {{
      color: var(--accent-emerald);
      font-size: 11px;
      font-weight: 600;
    }}

    .bar-track {{
      width: 100%;
      height: 14px;
      background: rgba(255, 255, 255, 0.05);
      border-radius: 4px;
      overflow: hidden;
      position: relative;
    }}

    .bar-fill {{
      height: 100%;
      border-radius: 4px;
      transition: width 0.4s ease-out;
    }}

    /* Topology Layers Flow */
    .topology-flow-card {{
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: 14px;
      padding: 18px 24px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }}

    .layers-container {{
      display: flex;
      align-items: center;
      gap: 14px;
      overflow-x: auto;
      padding: 6px 0;
    }}

    .dag-layer-box {{
      display: flex;
      flex-direction: column;
      gap: 8px;
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: 10px;
      padding: 10px 14px;
      min-width: 170px;
    }}

    .layer-title {{
      font-size: 11px;
      font-weight: 700;
      color: var(--text-dim);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}

    .layer-nodes-flow {{
      display: flex;
      flex-direction: column;
      gap: 6px;
    }}

    .layer-node-pill {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      padding: 4px 8px;
      border-radius: 6px;
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid var(--border-subtle);
      font-size: 12px;
    }}

    .layer-node-pill.is-crit {{
      border-color: rgba(244, 63, 94, 0.4);
      background: rgba(244, 63, 94, 0.08);
    }}

    .layer-node-id {{
      font-weight: 600;
      color: var(--accent-cyan);
    }}

    .layer-node-st {{
      font-size: 10px;
      text-transform: uppercase;
      color: var(--text-dim);
    }}

    .dag-layer-arrow {{
      color: var(--text-dim);
      font-size: 18px;
      user-select: none;
    }}

    /* Node Controls & Filters */
    .controls-bar {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 12px;
      padding: 12px 18px;
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: 12px;
    }}

    .filters-group {{
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }}

    .filter-btn {{
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      color: var(--text-muted);
      padding: 6px 12px;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.15s;
    }}

    .filter-btn:hover, .filter-btn.active {{
      background: var(--accent-cyan);
      color: #04101e;
      border-color: var(--accent-cyan);
    }}

    .search-box {{
      position: relative;
      min-width: 240px;
    }}

    .search-input {{
      width: 100%;
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      color: var(--text-main);
      padding: 6px 12px 6px 28px;
      border-radius: 6px;
      font-size: 12px;
      outline: none;
    }}

    .search-input:focus {{
      border-color: var(--accent-cyan);
    }}

    .search-icon {{
      position: absolute;
      left: 8px;
      top: 50%;
      transform: translateY(-50%);
      color: var(--text-dim);
      font-size: 12px;
    }}

    /* Nodes Grid */
    .nodes-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
      gap: 18px;
    }}

    .node-card {{
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: 14px;
      padding: 20px;
      display: flex;
      flex-direction: column;
      gap: 16px;
      transition: all 0.2s;
    }}

    .node-card:hover {{
      border-color: rgba(56, 189, 248, 0.4);
      box-shadow: 0 6px 24px rgba(0, 0, 0, 0.35);
    }}

    .node-header {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 12px;
    }}

    .node-title-group {{
      display: flex;
      flex-direction: column;
      gap: 4px;
    }}

    .node-id-row {{
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }}

    .node-id {{
      font-family: var(--font-mono);
      font-size: 14px;
      font-weight: 700;
      color: #ffffff;
    }}

    .badge-critical {{
      font-size: 10px;
      font-weight: 700;
      padding: 2px 6px;
      border-radius: 4px;
      background: rgba(244, 63, 94, 0.16);
      color: var(--accent-rose);
      border: 1px solid rgba(244, 63, 94, 0.3);
      text-transform: uppercase;
    }}

    .badge-bottleneck {{
      font-size: 10px;
      font-weight: 700;
      padding: 2px 6px;
      border-radius: 4px;
      background: rgba(251, 191, 36, 0.16);
      color: var(--accent-amber);
      border: 1px solid rgba(251, 191, 36, 0.3);
      text-transform: uppercase;
    }}

    .node-label {{
      font-size: 12px;
      color: var(--text-muted);
    }}

    .node-badges {{
      display: flex;
      align-items: center;
      gap: 6px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }}

    .badge-category {{
      font-size: 11px;
      font-weight: 600;
      padding: 3px 8px;
      border-radius: 6px;
      text-transform: uppercase;
    }}

    .badge-role {{
      font-size: 11px;
      font-weight: 600;
      padding: 3px 8px;
      border-radius: 6px;
      background: rgba(255, 255, 255, 0.06);
      color: var(--text-muted);
      border: 1px solid var(--border-subtle);
    }}

    .badge-status {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 11px;
      font-weight: 600;
      padding: 3px 8px;
      border-radius: 6px;
      border: 1px solid var(--border-subtle);
    }}

    .node-meta-grid {{
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 8px;
      background: var(--bg-surface);
      padding: 10px 14px;
      border-radius: 8px;
      border: 1px solid var(--border-subtle);
      font-size: 12px;
    }}

    .node-meta-item {{
      display: flex;
      flex-direction: column;
      gap: 2px;
    }}

    .meta-label {{
      color: var(--text-dim);
      font-size: 10px;
      text-transform: uppercase;
      font-weight: 600;
    }}

    .meta-value {{
      color: var(--text-main);
      font-size: 12px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}

    .token-strip {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(70px, 1fr));
      gap: 8px;
      padding: 10px;
      background: rgba(255, 255, 255, 0.02);
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
    }}

    .token-col {{
      display: flex;
      flex-direction: column;
      gap: 2px;
    }}

    .token-lbl {{
      font-size: 10px;
      color: var(--text-dim);
      text-transform: uppercase;
      font-weight: 600;
    }}

    .token-val {{
      font-size: 13px;
      font-weight: 700;
      color: #ffffff;
    }}

    .token-sub {{
      font-size: 9px;
      color: var(--text-dim);
    }}

    .node-cost-card {{
      background: rgba(16, 185, 129, 0.04);
      border: 1px solid rgba(16, 185, 129, 0.2);
      border-radius: 8px;
      padding: 10px 14px;
    }}

    .cost-row {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 8px;
    }}

    .cost-item {{
      display: flex;
      flex-direction: column;
      gap: 2px;
    }}

    .cost-sublabel {{
      font-size: 10px;
      color: var(--text-dim);
      text-transform: uppercase;
    }}

    .cost-figure {{
      font-size: 14px;
      font-weight: 700;
    }}

    .cost-vs {{
      color: var(--text-dim);
      font-size: 11px;
    }}

    .cost-savings-pill {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      background: rgba(16, 185, 129, 0.16);
      border: 1px solid rgba(16, 185, 129, 0.35);
      border-radius: 6px;
      padding: 4px 8px;
      color: var(--accent-emerald);
      font-size: 11px;
    }}

    .savings-val {{
      font-weight: 700;
    }}

    .node-footer {{
      display: flex;
      flex-direction: column;
      gap: 8px;
      padding-top: 10px;
      border-top: 1px solid var(--border-subtle);
      font-size: 11px;
    }}

    .node-footer-section {{
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }}

    .footer-sec-label {{
      color: var(--text-dim);
      min-width: 80px;
    }}

    .deps-container, .writes-container {{
      display: flex;
      align-items: center;
      gap: 6px;
      flex-wrap: wrap;
    }}

    .dep-pill {{
      font-family: var(--font-mono);
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      padding: 2px 6px;
      border-radius: 4px;
      color: var(--accent-cyan);
    }}

    .write-file-pill {{
      font-family: var(--font-mono);
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid var(--border-subtle);
      padding: 2px 6px;
      border-radius: 4px;
      color: var(--text-muted);
      max-width: 220px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}

    /* Raw JSON details drawer */
    .raw-details {{
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: 12px;
      padding: 16px;
    }}

    .raw-details summary {{
      cursor: pointer;
      font-weight: 600;
      color: var(--text-muted);
      user-select: none;
    }}

    .raw-details pre {{
      margin-top: 12px;
      background: #04070e;
      padding: 14px;
      border-radius: 8px;
      overflow-x: auto;
      font-family: var(--font-mono);
      font-size: 11px;
      color: #cbd5e1;
      border: 1px solid var(--border-subtle);
      max-height: 380px;
    }}

    /* Global Utilities */
    .font-mono {{ font-family: var(--font-mono); }}
    .text-cyan {{ color: var(--accent-cyan); }}
    .text-emerald {{ color: var(--accent-emerald); }}
    .text-indigo {{ color: var(--accent-indigo); }}
    .text-pink {{ color: var(--accent-pink); }}
    .text-muted {{ color: var(--text-muted); }}

    /* Responsive */
    @media (max-width: 768px) {{
      body {{ padding: 12px; }}
      .header-top {{ flex-direction: column; }}
      .kpi-grid {{ grid-template-columns: 1fr; }}
      .nodes-grid {{ grid-template-columns: 1fr; }}
      .controls-bar {{ flex-direction: column; align-items: stretch; }}
    }}
  </style>
</head>
<body>
  <div class="container">
    <!-- Header -->
    <header class="dashboard-header">
      <div class="header-top">
        <div class="title-area">
          <div class="brand-group">
            <span class="plugin-pill">⚡ DAG-ULW Monitor</span>
            <span class="preview-pill">Hermes Desktop ::preview</span>
          </div>
          <h1>{html.escape(name)}</h1>
          {f'<div class="goal-desc">{html.escape(goal)}</div>' if goal else ''}
        </div>
        <div class="header-badges">
          <div class="status-badge-lg" style="color: {st_color}; background: {st_bg};">
            <span class="status-dot {st_dot}"></span>
            <span>{st_label}</span>
          </div>
        </div>
      </div>

      <!-- OpenCodex Gateway summary bar -->
      <div class="gateway-summary-bar">
        <div class="gateway-left">
          <span class="gateway-pulse-pill">
            <span class="status-dot {'dot-running' if oc_avail else 'dot-pending'}"></span>
            OpenCodex Gateway: {'Active' if oc_avail else 'Standby'}
          </span>
          <span class="font-mono text-cyan">{oc_records} reqs captured</span>
        </div>
        <div class="gateway-details">
          <span>Providers: <strong class="text-main">{', '.join(html.escape(p) for p in oc_providers) or 'None'}</strong></span>
          <span>Models: <strong class="text-main">{', '.join(html.escape(m) for m in oc_models) or 'None'}</strong></span>
          <span>Gateway Duration: <strong class="font-mono text-main">{_fmt_duration(oc_duration_ms / 1000.0)}</strong></span>
          <span>Gateway Vol: <strong class="font-mono text-main">{_fmt_tokens(oc_total_tokens)} tok</strong></span>
        </div>
      </div>

      <div class="header-meta-bar">
        <div class="meta-pills-list">
          <div class="pill-info">Run ID: <strong class="font-mono">{html.escape(run_id)}</strong></div>
          <div class="pill-info">Nodes: <strong>{total_nodes}</strong> ({completed_nodes} done, {running_nodes} run, {pending_nodes} pend)</div>
          <div class="pill-info">Wall Time: <strong class="font-mono">{_fmt_duration(wall_duration_s)}</strong></div>
          {f'<div class="pill-info" title="{html.escape(session_id)}">Session: <strong class="font-mono">{html.escape(session_id[:16])}…</strong></div>' if session_id else ''}
        </div>
        <div class="time-meta">Updated: {_fmt_timestamp(updated_at)}</div>
      </div>
    </header>

    <!-- KPI Summary Grid -->
    <section class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-title">
          <span>Total Tokens</span>
          <span class="text-cyan font-mono">{api_calls} calls</span>
        </div>
        <div class="kpi-val font-mono">{_fmt_tokens(total_tok)}</div>
        <div class="kpi-subtext">
          <div class="kpi-breakdown-row">
            <span>Input Prompt</span>
            <span class="font-mono">{_fmt_tokens(inp_tok)}</span>
          </div>
          <div class="kpi-breakdown-row">
            <span>Output Completion</span>
            <span class="font-mono">{_fmt_tokens(out_tok)}</span>
          </div>
          <div class="kpi-breakdown-row">
            <span>Cache Read</span>
            <span class="font-mono text-indigo">{_fmt_tokens(cache_tok)}</span>
          </div>
          {f'''<div class="kpi-breakdown-row">
            <span>Reasoning</span>
            <span class="font-mono text-pink">{_fmt_tokens(reason_tok)}</span>
          </div>''' if reason_tok > 0 else ''}
        </div>
      </div>

      <div class="kpi-card">
        <div class="kpi-title">
          <span>Cache Hit Rate</span>
          <span class="text-indigo font-mono">Efficiency</span>
        </div>
        <div class="kpi-val font-mono" style="color: #818cf8;">{cache_hit_pct}%</div>
        <div class="kpi-subtext">
          <span>{_fmt_tokens(cache_tok)} cached / {_fmt_tokens(cache_denominator)} prompt tokens</span>
          <div class="progress-bar-wrap">
            <div class="progress-bar-fill" style="width: {min(100.0, cache_hit_pct)}%;"></div>
          </div>
        </div>
      </div>

      <div class="kpi-card">
        <div class="kpi-title">
          <span>Estimated Execution Cost</span>
          <span class="text-muted font-mono">Runtime</span>
        </div>
        <div class="kpi-val font-mono">{_fmt_usd(gemini_cost, 4)}</div>
        <div class="kpi-subtext">
          <div class="kpi-breakdown-row">
            <span>Opus 4.6 Baseline</span>
            <span class="font-mono">{_fmt_usd(opus_cost, 4)}</span>
          </div>
          <div class="kpi-breakdown-row">
            <span>Sonnet 4.6 Baseline</span>
            <span class="font-mono">{_fmt_usd(sonnet_cost, 4)}</span>
          </div>
        </div>
      </div>

      <div class="kpi-card highlight-savings">
        <div class="kpi-title">
          <span>Net Cost Savings</span>
          <span class="text-emerald font-mono">vs Opus 4.6</span>
        </div>
        <div class="kpi-val text-emerald font-mono">+{_fmt_usd(net_saved_usd, 4)}</div>
        <div class="kpi-subtext">
          <div class="kpi-breakdown-row">
            <span>Relative Savings</span>
            <span class="font-mono text-emerald" style="font-weight:700;">{net_saved_pct:.1f}%</span>
          </div>
          <div class="kpi-breakdown-row">
            <span>Efficiency Multiplier</span>
            <span class="font-mono text-emerald" style="font-weight:700;">{cost_multiplier:.1f}x cheaper</span>
          </div>
        </div>
      </div>
    </section>

    <!-- Multi-Model Benchmark Comparison -->
    <section class="comparison-section">
      <div class="section-header-row">
        <div class="section-title">
          <span>Multi-Model Cost Benchmark</span>
          <span class="section-badge">{cost_multiplier:.1f}x Savings vs Frontier</span>
        </div>
        <div class="text-muted" style="font-size: 12px;">Standardized USD rate card based on {_fmt_tokens(total_tok)} tokens</div>
      </div>
      <div class="benchmark-list">
        {''.join(comparison_bars_html)}
      </div>
    </section>

    <!-- Topology Layers Pipeline -->
    <section class="topology-flow-card">
      <div class="section-header-row">
        <div class="section-title">
          <span>DAG Execution Topology & Layers</span>
          {f'<span class="badge-critical">Critical Path: {" → ".join(critical_path)}</span>' if critical_path else ''}
        </div>
      </div>
      <div class="layers-container">
        {layers_flow_html}
      </div>
    </section>

    <!-- Filter & Search Controls -->
    <section class="controls-bar">
      <div class="filters-group">
        <button class="filter-btn active" data-filter="all">All ({total_nodes})</button>
        <button class="filter-btn" data-filter="completed">Completed ({completed_nodes})</button>
        <button class="filter-btn" data-filter="running">Running ({running_nodes})</button>
        <button class="filter-btn" data-filter="pending">Pending ({pending_nodes})</button>
      </div>
      <div class="search-box">
        <span class="search-icon">🔍</span>
        <input type="text" id="nodeSearch" class="search-input" placeholder="Search node by ID or label…">
      </div>
    </section>

    <!-- Node Cards Grid -->
    <section class="nodes-grid" id="nodesContainer">
      {''.join(node_cards_html)}
    </section>

    <!-- Raw Telemetry Inspector -->
    <details class="raw-details">
      <summary>View Raw DAG Telemetry JSON</summary>
      <pre><code>{html.escape(json.dumps(telemetry_data, indent=2))}</code></pre>
    </details>
  </div>

  <script>
    // Client-side interactive filtering and search
    (function() {{
      const filterBtns = document.querySelectorAll('.filter-btn');
      const searchInput = document.getElementById('nodeSearch');
      const nodeCards = document.querySelectorAll('.node-card');

      let currentFilter = 'all';
      let currentQuery = '';

      function updateVisibility() {{
        nodeCards.forEach(card => {{
          const status = card.getAttribute('data-status') || '';
          const nodeId = (card.getAttribute('data-node-id') || '').toLowerCase();
          const label = (card.querySelector('.node-label')?.textContent || '').toLowerCase();

          const matchesStatus = (currentFilter === 'all') || (status === currentFilter);
          const matchesQuery = !currentQuery || nodeId.includes(currentQuery) || label.includes(currentQuery);

          if (matchesStatus && matchesQuery) {{
            card.style.display = 'flex';
          }} else {{
            card.style.display = 'none';
          }}
        }});
      }}

      filterBtns.forEach(btn => {{
        btn.addEventListener('click', () => {{
          filterBtns.forEach(b => b.classList.remove('active'));
          btn.classList.add('active');
          currentFilter = btn.getAttribute('data-filter') || 'all';
          updateVisibility();
        }});
      }});

      if (searchInput) {{
        searchInput.addEventListener('input', (e) => {{
          currentQuery = (e.target.value || '').trim().toLowerCase();
          updateVisibility();
        }});
      }}
    }})();
  </script>
</body>
</html>
"""


def save_dashboard_html(
    telemetry_data: dict[str, Any],
    output_path: Optional[str | Path] = None,
) -> str:
    """Render and save DAG telemetry dashboard HTML to scratch or requested path.

    Parameters:
        telemetry_data: Dictionary returned by `telemetry.get_dag_telemetry()`.
        output_path: Optional path. Defaults to `/root/.hermes/cache/scratch/dag-monitor-{run_id}.html`.

    Returns:
        Absolute string path to the saved HTML file.
    """
    if not output_path:
        raw_run_id = str(telemetry_data.get("run_id") or "latest")
        clean_run_id = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", raw_run_id)
        target = DEFAULT_SCRATCH_DIR / f"dag-monitor-{clean_run_id}.html"
    else:
        target = Path(output_path)

    # Ensure parent directory exists
    target.parent.mkdir(parents=True, exist_ok=True)

    html_content = render_dag_dashboard(telemetry_data)

    target.write_text(html_content, encoding="utf-8")
    return str(target.resolve())
