"""`dag_monitor.telemetry`: Telemetry extraction for DAG runs, tokens, and model costs.

Extracts DAG run structure from OMH runtime records, links subagent sessions
from `state.db`, parses granular LLM request metrics from `usage.jsonl`,
and computes relative cost benchmarks and savings against reference frontier models
(e.g. Claude Opus 4.6 vs Gemini 3.8 Flash vs DeepSeek Flash).
"""

from __future__ import annotations

import glob
import json
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


# --- Pricing Rate Cards (USD per 1,000,000 tokens) --------------------------
# Baseline reference: Claude Opus 4.6 ($5 in, $25 out, $0.50 cache read)
# Standard comparison suite across major model classes.
MODEL_PRICING: dict[str, dict[str, Any]] = {
    "claude-opus-4-6": {
        "label": "Claude Opus 4.6",
        "provider": "anthropic",
        "input_per_m": 5.00,
        "output_per_m": 25.00,
        "cache_read_per_m": 0.50,
        "cache_write_per_m": 6.25,
    },
    "claude-sonnet-4-6": {
        "label": "Claude Sonnet 4.6",
        "provider": "anthropic",
        "input_per_m": 3.00,
        "output_per_m": 15.00,
        "cache_read_per_m": 0.30,
        "cache_write_per_m": 3.75,
    },
    "gemini-3.8-flash": {
        "label": "Gemini 3.8 Flash",
        "provider": "google",
        "input_per_m": 0.75,
        "output_per_m": 3.75,
        "cache_read_per_m": 0.075,
        "cache_write_per_m": 0.0,
    },
    "deepseek-flash": {
        "label": "DeepSeek Flash",
        "provider": "deepseek",
        "input_per_m": 0.15,
        "output_per_m": 0.60,
        "cache_read_per_m": 0.003,
        "cache_write_per_m": 0.0,
    },
    "gpt-5.6-sol": {
        "label": "GPT-5.6 Sol",
        "provider": "openai",
        "input_per_m": 5.00,
        "output_per_m": 30.00,
        "cache_read_per_m": 0.50,
        "cache_write_per_m": 6.25,
    },
}


# --- Environment & Path Resolvers -------------------------------------------
def get_hermes_home() -> Path:
    """Return Hermes home directory from env HERMES_HOME or ~/.hermes default."""
    return Path(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")))


def get_omh_home() -> Path:
    """Return OMH home directory from env OMH_HOME or ~/.omh default."""
    return Path(os.environ.get("OMH_HOME", os.path.expanduser("~/.omh")))


def get_default_runtime_paths() -> tuple[Path, ...]:
    """Return runtime search directories for DAG execution records."""
    omh = get_omh_home()
    return (
        omh / "runtime" / "dag",
        omh / "state" / "runs",
    )


def get_default_state_db() -> Path:
    """Return primary Hermes state.db path."""
    return get_hermes_home() / "state.db"


def get_default_usage_jsonl() -> Path:
    """Return OpenCodex usage log path from env OPENCODEX_USAGE_PATH or ~/.opencodex/usage.jsonl."""
    return Path(os.environ.get("OPENCODEX_USAGE_PATH", os.path.expanduser("~/.opencodex/usage.jsonl")))


DEFAULT_RUNTIME_PATHS = get_default_runtime_paths()
DEFAULT_STATE_DB = get_default_state_db()
DEFAULT_USAGE_JSONL = get_default_usage_jsonl()


def compute_token_costs(
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
) -> dict[str, dict[str, float]]:
    """Compute estimated USD costs across all known model rate cards."""
    costs: dict[str, dict[str, float]] = {}
    for model_key, card in MODEL_PRICING.items():
        inp = (input_tokens * card["input_per_m"]) / 1_000_000.0
        out = (output_tokens * card["output_per_m"]) / 1_000_000.0
        cache = (cache_read_tokens * card["cache_read_per_m"]) / 1_000_000.0
        total = inp + out + cache
        costs[model_key] = {
            "input_cost_usd": round(inp, 6),
            "output_cost_usd": round(out, 6),
            "cache_cost_usd": round(cache, 6),
            "total_cost_usd": round(total, 6),
        }
    return costs


def compute_savings_summary(
    costs_dict: dict[str, dict[str, float]],
    baseline_key: str = "claude-opus-4-6",
) -> dict[str, Any]:
    """Calculate relative savings of commodity/flash models compared to the frontier baseline."""
    base_cost = costs_dict.get(baseline_key, {}).get("total_cost_usd", 0.0)
    savings: dict[str, Any] = {}

    for comp_key in ("gemini-3.8-flash", "deepseek-flash", "claude-sonnet-4-6"):
        if comp_key not in costs_dict:
            continue
        comp_cost = costs_dict[comp_key].get("total_cost_usd", 0.0)
        saved_usd = max(0.0, base_cost - comp_cost)
        pct = (saved_usd / base_cost * 100.0) if base_cost > 0 else 0.0
        multiplier = (base_cost / comp_cost) if comp_cost > 0 else 1.0
        savings[comp_key] = {
            "baseline_model": baseline_key,
            "compared_model": comp_key,
            "baseline_cost_usd": round(base_cost, 6),
            "compared_cost_usd": round(comp_cost, 6),
            "saved_usd": round(saved_usd, 6),
            "saved_pct": round(pct, 2),
            "multiplier": round(multiplier, 2),
            "summary_label": f"{multiplier:.1f}x cheaper ({pct:.1f}% savings vs {baseline_key})",
        }
    return savings


def find_dag_runs(
    session_id: Optional[str] = None,
    run_id: Optional[str] = None,
    search_dirs: Optional[Sequence[Path]] = None,
) -> list[tuple[dict[str, Any], Path]]:
    """Scan runtime directories and return matching DAG runs sorted newest first."""
    candidates: list[tuple[dict[str, Any], Path]] = []
    dirs_to_search = search_dirs if search_dirs is not None else get_default_runtime_paths()

    for base_dir in dirs_to_search:
        if not base_dir.is_dir():
            continue
        for root, _, files in os.walk(base_dir):
            for file_name in files:
                if not file_name.endswith(".json") or file_name.startswith("."):
                    continue
                file_path = Path(root) / file_name
                try:
                    data = json.loads(file_path.read_text(encoding="utf-8"))
                    if not isinstance(data, dict):
                        continue
                    # Must be a DAG record (either has 'nodes' or schema_version starts with 'omh_dag')
                    if "nodes" not in data and not str(data.get("schema_version", "")).startswith("omh_dag"):
                        continue

                    r_id = data.get("run") or data.get("id") or file_path.stem
                    s_id = data.get("session_id") or ""

                    # Filter by run_id if provided
                    if run_id:
                        if r_id != run_id and file_path.stem != run_id and run_id not in str(file_path):
                            continue

                    # Filter by session_id if provided
                    if session_id:
                        if session_id not in s_id and session_id not in str(file_path):
                            continue

                    candidates.append((data, file_path))
                except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                    continue

    def sort_key(item: tuple[dict[str, Any], Path]) -> float:
        data, path = item
        for field in ("updated_at", "defined_at", "created_at"):
            val = data.get(field)
            if isinstance(val, (int, float)):
                return float(val)
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    candidates.sort(key=sort_key, reverse=True)
    return candidates


def query_state_db(
    run_id: str,
    node_ids: Sequence[str],
    parent_session_id: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> dict[str, dict[str, Any]]:
    """Query Hermes state.db to find matching subagent sessions for each DAG node."""
    target_db = db_path if db_path is not None else get_default_state_db()
    results: dict[str, dict[str, Any]] = {node_id: {} for node_id in node_ids}
    if not target_db.is_file():
        return results

    try:
        conn = sqlite3.connect(f"file:{target_db}?mode=ro", uri=True, timeout=10.0)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Step A: Find node mappings from messages table (user message has `[dag:<run>/<node>]`)
        node_session_map: dict[str, str] = {}
        cursor.execute(
            """
            SELECT session_id, content
            FROM messages
            WHERE role = 'user' AND content LIKE ?
            ORDER BY timestamp ASC
            """,
            (f"%[dag:{run_id}/%",),
        )
        marker_pattern = re.compile(rf"\[dag:{re.escape(run_id)}/([a-zA-Z0-9_-]+)\]")
        for row in cursor.fetchall():
            session_id = row["session_id"]
            content = row["content"] or ""
            match = marker_pattern.search(content)
            if match:
                node_name = match.group(1)
                if node_name in node_ids:
                    node_session_map[node_name] = session_id

        # Step B: Also check async_delegations table
        cursor.execute(
            """
            SELECT delegation_id, parent_session_id, task_json, dispatched_at, completed_at
            FROM async_delegations
            WHERE task_json LIKE ?
            ORDER BY dispatched_at ASC
            """,
            (f"%[dag:{run_id}/%",),
        )
        for row in cursor.fetchall():
            task_raw = row["task_json"] or ""
            match = marker_pattern.search(task_raw)
            if match:
                node_name = match.group(1)
                # If we don't have a session_id mapped yet, check child sessions under this parent
                if node_name in node_ids and node_name not in node_session_map:
                    p_id = row["parent_session_id"]
                    cursor.execute(
                        """
                        SELECT id FROM sessions
                        WHERE parent_session_id = ? AND started_at >= ? - 5.0
                        ORDER BY started_at ASC LIMIT 1
                        """,
                        (p_id, row["dispatched_at"]),
                    )
                    child = cursor.fetchone()
                    if child:
                        node_session_map[node_name] = child["id"]

        # Step C: Fallback check in sessions table for any unmapped nodes
        for node_id in node_ids:
            if node_id not in node_session_map:
                cursor.execute(
                    """
                    SELECT id FROM sessions
                    WHERE title LIKE ? OR session_key LIKE ?
                    ORDER BY started_at DESC LIMIT 1
                    """,
                    (f"%[dag:{run_id}/{node_id}]%", f"%[dag:{run_id}/{node_id}]%"),
                )
                cand = cursor.fetchone()
                if cand:
                    node_session_map[node_id] = cand["id"]

        # Step D: Retrieve session records and populate tokens from sessions and session_model_usage
        for node_id, s_id in node_session_map.items():
            cursor.execute(
                """
                SELECT id, parent_session_id, model, started_at, ended_at,
                       input_tokens, output_tokens, cache_read_tokens, cache_write_tokens,
                       reasoning_tokens, api_call_count, billing_provider, title
                FROM sessions
                WHERE id = ?
                """,
                (s_id,),
            )
            session_row = cursor.fetchone()
            if not session_row:
                continue

            # Query session_model_usage for granular per-model telemetry
            cursor.execute(
                """
                SELECT model, billing_provider, api_call_count, input_tokens, output_tokens,
                       cache_read_tokens, cache_write_tokens, reasoning_tokens,
                       estimated_cost_usd, actual_cost_usd
                FROM session_model_usage
                WHERE session_id = ?
                """,
                (s_id,),
            )
            model_usages = [dict(r) for r in cursor.fetchall()]

            inp_tokens = session_row["input_tokens"] or 0
            out_tokens = session_row["output_tokens"] or 0
            cache_read_tokens = session_row["cache_read_tokens"] or 0
            cache_write_tokens = session_row["cache_write_tokens"] or 0
            reasoning_tokens = session_row["reasoning_tokens"] or 0
            api_call_count = session_row["api_call_count"] or 0

            # Fallback/refinement from session_model_usage if present
            if model_usages:
                mu_inp = sum(m.get("input_tokens") or 0 for m in model_usages)
                mu_out = sum(m.get("output_tokens") or 0 for m in model_usages)
                mu_cread = sum(m.get("cache_read_tokens") or 0 for m in model_usages)
                mu_cwrite = sum(m.get("cache_write_tokens") or 0 for m in model_usages)
                mu_reason = sum(m.get("reasoning_tokens") or 0 for m in model_usages)
                mu_calls = sum(m.get("api_call_count") or 0 for m in model_usages)

                inp_tokens = max(inp_tokens, mu_inp)
                out_tokens = max(out_tokens, mu_out)
                cache_read_tokens = max(cache_read_tokens, mu_cread)
                cache_write_tokens = max(cache_write_tokens, mu_cwrite)
                reasoning_tokens = max(reasoning_tokens, mu_reason)
                api_call_count = max(api_call_count, mu_calls)

            model_name = session_row["model"] or (
                model_usages[0]["model"] if model_usages and model_usages[0].get("model") else "unknown"
            )
            provider_name = session_row["billing_provider"] or (
                model_usages[0]["billing_provider"] if model_usages and model_usages[0].get("billing_provider") else "custom"
            )

            results[node_id] = {
                "session_id": session_row["id"],
                "parent_session_id": session_row["parent_session_id"],
                "model": model_name,
                "billing_provider": provider_name,
                "started_at": session_row["started_at"],
                "ended_at": session_row["ended_at"],
                "input_tokens": inp_tokens,
                "output_tokens": out_tokens,
                "cache_read_tokens": cache_read_tokens,
                "cache_write_tokens": cache_write_tokens,
                "reasoning_tokens": reasoning_tokens,
                "api_call_count": api_call_count,
                "title": session_row["title"],
                "model_usage_breakdown": model_usages,
            }

        conn.close()
    except Exception:
        pass

    return results


def _binary_search_offset(file_path: Path, target_ts_ms: int) -> int:
    """Binary search byte offset in usage.jsonl corresponding to target_ts_ms."""
    try:
        size = file_path.stat().st_size
    except OSError:
        return 0

    low = 0
    high = size
    with open(file_path, "rb") as f:
        while low < high:
            mid = (low + high) // 2
            f.seek(mid)
            f.readline()  # skip partial line
            line = f.readline()
            if not line:
                high = mid
                continue
            try:
                data = json.loads(line.decode("utf-8", errors="ignore"))
                ts = data.get("timestamp", 0)
                if ts < target_ts_ms:
                    low = mid + 1
                else:
                    high = mid
            except Exception:
                low = mid + 1
    return max(0, low - 8192)


def inspect_opencodex_usage(
    since_ts: float,
    until_ts: Optional[float] = None,
    usage_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Inspect OpenCodex usage.jsonl for exact prompt/completion/cached tokens and provider."""
    result: dict[str, Any] = {
        "available": False,
        "records_count": 0,
        "providers": [],
        "models": [],
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
        "total_duration_ms": 0,
        "records_summary": [],
    }

    target_path = usage_path if usage_path is not None else get_default_usage_jsonl()
    if not target_path.is_file():
        return result

    try:
        if target_path.stat().st_size == 0:
            return result
    except OSError:
        return result

    since_ms = int((since_ts - 5.0) * 1000)
    until_ms = int((until_ts + 5.0) * 1000) if until_ts else None

    offset = _binary_search_offset(target_path, since_ms)

    providers_set: set[str] = set()
    models_set: set[str] = set()

    try:
        with open(target_path, "rb") as f:
            f.seek(offset)
            f.readline()  # align boundary
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line.decode("utf-8", errors="ignore"))
                    ts = obj.get("timestamp", 0)
                    if ts < since_ms:
                        continue
                    if until_ms and ts > until_ms:
                        break

                    usage = obj.get("usage", {}) or {}
                    inp = usage.get("inputTokens", 0)
                    out = usage.get("outputTokens", 0)
                    cached = usage.get("cachedInputTokens", 0)
                    reasoning = usage.get("reasoningOutputTokens", 0)
                    dur = obj.get("durationMs", 0)

                    result["records_count"] += 1
                    providers_set.add(obj.get("provider", "unknown"))
                    models_set.add(obj.get("model", "unknown"))
                    result["input_tokens"] += inp
                    result["output_tokens"] += out
                    result["cached_tokens"] += cached
                    result["reasoning_tokens"] += reasoning
                    result["total_tokens"] += obj.get("totalTokens", inp + out)
                    result["total_duration_ms"] += dur

                    # Keep lightweight sample records (first 3 and last 3)
                    if len(result["records_summary"]) < 10:
                        result["records_summary"].append({
                            "timestamp": ts,
                            "provider": obj.get("provider"),
                            "model": obj.get("model"),
                            "input_tokens": inp,
                            "output_tokens": out,
                            "cached_tokens": cached,
                            "duration_ms": dur,
                        })
                except Exception:
                    continue

        if result["records_count"] > 0:
            result["available"] = True
            result["providers"] = sorted(list(providers_set))
            result["models"] = sorted(list(models_set))
        else:
            result["available"] = False
            result["providers"] = []
            result["models"] = []
    except OSError:
        result["available"] = False
        result["providers"] = []
        result["models"] = []

    return result


def get_dag_telemetry(
    session_id: Optional[str] = None,
    run_id: Optional[str] = None,
    search_dirs: Optional[Sequence[Path]] = None,
    state_db_path: Optional[Path] = None,
    usage_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Extract DAG telemetry, token consumption from state.db & usage.jsonl, and savings benchmarks.

    Args:
        session_id: Optional session identifier or conversation key.
        run_id: Optional DAG run ID (e.g. 'dag-monitor-init').
        search_dirs: Optional search directories for DAG run records.
        state_db_path: Optional path to Hermes state.db.
        usage_path: Optional path to OpenCodex usage.jsonl.

    Returns:
        Structured dictionary containing run metadata, summary metrics, per-node
        token and duration breakdown, and relative savings calculations against Opus 4.6.
    """
    runs = find_dag_runs(session_id=session_id, run_id=run_id, search_dirs=search_dirs)
    if not runs:
        return {
            "error": "No matching DAG run found",
            "session_id": session_id,
            "run_id": run_id,
            "timestamp": time.time(),
        }

    record, record_path = runs[0]
    actual_run_id = record.get("run") or record.get("id") or record_path.stem
    actual_session_id = record.get("session_id") or session_id or ""
    defined_at = float(record.get("defined_at") or record_path.stat().st_mtime)
    updated_at = float(record.get("updated_at") or time.time())
    cancelled_at = record.get("cancelled_at")

    raw_nodes = record.get("nodes", [])
    node_ids = [str(node.get("id")) for node in raw_nodes if isinstance(node, dict)]

    target_db = state_db_path if state_db_path is not None else get_default_state_db()
    target_usage = usage_path if usage_path is not None else get_default_usage_jsonl()

    # Query state.db for child sessions (primary source for DAG subagents)
    db_sessions = query_state_db(actual_run_id, node_ids, parent_session_id=actual_session_id, db_path=target_db)

    # Inspect usage.jsonl (optional gateway layer)
    opencodex_data = inspect_opencodex_usage(since_ts=defined_at, usage_path=target_usage)

    nodes_telemetry: dict[str, Any] = {}
    tot_input = 0
    tot_output = 0
    tot_cache = 0
    tot_reasoning = 0
    tot_calls = 0

    completed_count = 0
    running_count = 0
    ready_count = 0
    pending_count = 0
    failed_count = 0

    for node in raw_nodes:
        n_id = str(node.get("id"))
        label = node.get("label") or n_id
        category = node.get("category", "general")
        role = node.get("role", "build")
        depends_on = node.get("depends_on", [])
        write_scope = node.get("write_scope", [])
        attempts = node.get("attempts", [])

        # Session db metrics
        s_data = db_sessions.get(n_id, {})
        child_session_id = s_data.get("session_id")
        child_model = s_data.get("model") or "agy-gemini-3.8-flash"
        child_provider = s_data.get("billing_provider") or "custom"

        inp_tokens = s_data.get("input_tokens", 0)
        out_tokens = s_data.get("output_tokens", 0)
        cache_tokens = s_data.get("cache_read_tokens", 0)
        reasoning_tokens = s_data.get("reasoning_tokens", 0)
        api_calls = s_data.get("api_call_count", 0)

        tot_input += inp_tokens
        tot_output += out_tokens
        tot_cache += cache_tokens
        tot_reasoning += reasoning_tokens
        tot_calls += api_calls

        # Determine node state & timings
        last_attempt = attempts[-1] if attempts else None
        start_time = None
        stop_time = None

        if last_attempt:
            start_time = last_attempt.get("handed_out_at")
            stop_time = last_attempt.get("settled_at")

        if start_time is None and s_data.get("started_at"):
            start_time = s_data.get("started_at")
        if stop_time is None and s_data.get("ended_at"):
            stop_time = s_data.get("ended_at")

        now = time.time()
        duration_s = None
        if start_time:
            end_t = stop_time or now
            duration_s = max(0.0, round(end_t - start_time, 2))

        # Status derivation
        if cancelled_at is not None:
            node_status = "cancelled"
        elif last_attempt and last_attempt.get("outcome") in ("accepted", "done", "success"):
            node_status = "completed"
            completed_count += 1
        elif last_attempt and last_attempt.get("outcome") in ("failed", "blocked"):
            node_status = "failed"
            failed_count += 1
        elif last_attempt and last_attempt.get("handed_out_at") is not None:
            if s_data.get("ended_at"):
                node_status = "returned"
                completed_count += 1
            else:
                node_status = "running"
                running_count += 1
        else:
            # Check dependencies
            deps_satisfied = all(
                nodes_telemetry.get(dep, {}).get("status") in ("completed", "accepted")
                for dep in depends_on
            ) if depends_on else True

            if deps_satisfied:
                node_status = "ready"
                ready_count += 1
            else:
                node_status = "pending"
                pending_count += 1

        # Calculate costs for this node
        node_costs = compute_token_costs(inp_tokens, out_tokens, cache_tokens)
        node_savings = compute_savings_summary(node_costs, baseline_key="claude-opus-4-6")

        nodes_telemetry[n_id] = {
            "id": n_id,
            "label": label,
            "category": category,
            "role": role,
            "status": node_status,
            "depends_on": depends_on,
            "write_scope": write_scope,
            "attempts_count": len(attempts),
            "start_time": start_time,
            "stop_time": stop_time,
            "duration_s": duration_s,
            "session_id": child_session_id,
            "model": child_model,
            "provider": child_provider,
            "tokens": {
                "input_tokens": inp_tokens,
                "output_tokens": out_tokens,
                "cache_read_tokens": cache_tokens,
                "reasoning_tokens": reasoning_tokens,
                "total_tokens": inp_tokens + out_tokens + cache_tokens,
                "api_call_count": api_calls,
            },
            "costs_usd": {
                k: v["total_cost_usd"] for k, v in node_costs.items()
            },
            "savings_vs_opus": node_savings.get("gemini-3.8-flash", {}),
        }

    # Run-level overall state
    if cancelled_at is not None:
        run_status = "cancelled"
    elif completed_count == len(raw_nodes) and len(raw_nodes) > 0:
        run_status = "completed"
    elif failed_count > 0:
        run_status = "failed"
    elif running_count > 0 or ready_count > 0:
        run_status = "running"
    else:
        run_status = "pending"

    # Total costs and savings across entire DAG
    total_costs = compute_token_costs(tot_input, tot_output, tot_cache)
    total_savings = compute_savings_summary(total_costs, baseline_key="claude-opus-4-6")

    # Coordinator session summary if available in state.db
    coordinator_summary = None
    if actual_session_id and target_db.is_file():
        try:
            conn = sqlite3.connect(f"file:{target_db}?mode=ro", uri=True, timeout=5.0)
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            c.execute(
                """
                SELECT id, model, input_tokens, output_tokens, cache_read_tokens,
                       reasoning_tokens, api_call_count
                FROM sessions
                WHERE id = ? OR session_key LIKE ?
                LIMIT 1
                """,
                (actual_session_id, f"%{actual_session_id}%"),
            )
            crow = c.fetchone()
            if crow:
                c_inp = crow["input_tokens"] or 0
                c_out = crow["output_tokens"] or 0
                c_cread = crow["cache_read_tokens"] or 0
                c_reason = crow["reasoning_tokens"] or 0
                c_calls = crow["api_call_count"] or 0

                # Check session_model_usage as fallback / refinement
                c.execute(
                    """
                    SELECT input_tokens, output_tokens, cache_read_tokens, reasoning_tokens, api_call_count
                    FROM session_model_usage
                    WHERE session_id = ?
                    """,
                    (crow["id"],),
                )
                c_mus = [dict(r) for r in c.fetchall()]
                if c_mus:
                    c_inp = max(c_inp, sum(m.get("input_tokens") or 0 for m in c_mus))
                    c_out = max(c_out, sum(m.get("output_tokens") or 0 for m in c_mus))
                    c_cread = max(c_cread, sum(m.get("cache_read_tokens") or 0 for m in c_mus))
                    c_reason = max(c_reason, sum(m.get("reasoning_tokens") or 0 for m in c_mus))
                    c_calls = max(c_calls, sum(m.get("api_call_count") or 0 for m in c_mus))

                coordinator_summary = {
                    "session_id": crow["id"],
                    "model": crow["model"] or "unknown",
                    "input_tokens": c_inp,
                    "output_tokens": c_out,
                    "cache_read_tokens": c_cread,
                    "reasoning_tokens": c_reason,
                    "api_call_count": c_calls,
                }
            conn.close()
        except Exception:
            pass

    return {
        "run_id": actual_run_id,
        "name": record.get("name", actual_run_id),
        "goal": record.get("goal", ""),
        "session_id": actual_session_id,
        "status": run_status,
        "defined_at": defined_at,
        "updated_at": updated_at,
        "record_path": str(record_path),
        "topology": record.get("topology", {}),
        "summary": {
            "total_nodes": len(raw_nodes),
            "completed_nodes": completed_count,
            "running_nodes": running_count,
            "ready_nodes": ready_count,
            "pending_nodes": pending_count,
            "failed_nodes": failed_count,
            "tokens": {
                "input_tokens": tot_input,
                "output_tokens": tot_output,
                "cache_read_tokens": tot_cache,
                "reasoning_tokens": tot_reasoning,
                "total_tokens": tot_input + tot_output + tot_cache,
                "api_call_count": tot_calls,
            },
            "costs_usd": {
                k: v["total_cost_usd"] for k, v in total_costs.items()
            },
            "cost_breakdown": total_costs,
            "savings": total_savings,
        },
        "nodes": nodes_telemetry,
        "coordinator_session": coordinator_summary,
        "opencodex_telemetry": {
            "available": bool(opencodex_data.get("available", False)),
            "records_count": int(opencodex_data.get("records_count", 0)),
            "providers": list(opencodex_data.get("providers", [])),
            "models": list(opencodex_data.get("models", [])),
            "tokens": {
                "input_tokens": int(opencodex_data.get("input_tokens", 0)),
                "output_tokens": int(opencodex_data.get("output_tokens", 0)),
                "cached_tokens": int(opencodex_data.get("cached_tokens", 0)),
                "reasoning_tokens": int(opencodex_data.get("reasoning_tokens", 0)),
                "total_tokens": int(opencodex_data.get("total_tokens", 0)),
            },
            "total_duration_ms": int(opencodex_data.get("total_duration_ms", 0)),
        },
    }


if __name__ == "__main__":
    import pprint
    res = get_dag_telemetry()
    pprint.pprint(res)
