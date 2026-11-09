# hermes-dag-monitor

> Visual token telemetry, multi-model cost benchmarks, and quota savings for Hermes Agent and DAG-ULW workflows.

`hermes-dag-monitor` is an in-session telemetry and dashboard plugin for [Hermes Agent](https://github.com/NousResearch/hermes-agent). It aggregates real-time token metrics, prompt cache utilization, and execution costs across all parallel and sequential DAG execution nodes, displaying both an in-chat summary and an interactive standalone HTML visual dashboard.

---

## Features

- **Per-Node Token & Cost Telemetry**: Granular breakdown of input tokens, output tokens, prompt cache reads, and latency across each DAG node.
- **Multi-Model Cost Benchmarks**: Automated comparative savings benchmark against high-end frontier models (such as Claude 3.5 Sonnet and Claude Opus 4.6), detailing monetary savings and multiplier efficiency.
- **Prompt Cache Hit Tracking**: Clear visibility into cache read efficiency, demonstrating reuse across iterative agent cycles.
- **Zero External Dependencies**: Implemented strictly with standard library Python and inline CSS/SVG/JS—no third-party dependencies required.
- **Responsive Dark Theme UI**: Polished standalone HTML dashboard with node cards, visual status indicators (completed, running, failed), summary statistics, and interactive preview integration (`::preview`).

---

## Installation

### Via Hermes Plugins CLI

Install directly from Git repository or owner/repo:

```bash
hermes plugins add https://github.com/<owner>/hermes-dag-monitor
# or using install
hermes plugins install <owner>/hermes-dag-monitor
```

### Manual Installation (Symlink or Copy)

Clone or link the repository directly into your active Hermes plugins directory:

```bash
# Symlink into Hermes plugins
ln -s /path/to/hermes-dag-monitor ~/.hermes/plugins/dag-monitor

# Or copy into plugins
mkdir -p ~/.hermes/plugins/dag-monitor
cp -r /path/to/hermes-dag-monitor/* ~/.hermes/plugins/dag-monitor/
```

Verify plugin registration and runtime integrity:

```bash
hermes plugins doctor ~/.hermes/plugins/dag-monitor
```

---

## Usage

In any active Hermes Agent session, invoke the slash command:

```text
/dag-monitor
```

You can also inspect a specific DAG run by passing the run ID:

```text
/dag-monitor 20261108_202502_87ee5b-0d7751bfefe488ed
```

### Output

The command emits:
1. An inline markdown summary with node progress, total tokens, prompt cache hit percentage, and savings relative to Claude Opus 4.6.
2. A rich visual dashboard preview tag (`::preview{file="..."}`) rendering the interactive dark-themed report.

---

## Architecture

- **`plugin.yaml`**: Native Hermes Agent plugin manifest defining commands and metadata.
- **`__init__.py`**: Plugin lifecycle and `/dag-monitor` slash command handler.
- **`telemetry.py`**: Telemetry engine collecting token counts, model usage, and timing from Hermes `state.db` and DAG runtime manifests.
- **`render.py`**: Standalone HTML and markdown template generator.

---

## License

MIT License © 2026 Rafael Moreira
