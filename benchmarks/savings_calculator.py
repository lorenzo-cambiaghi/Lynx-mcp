"""Translate Lynx's *measured* token savings into money, at today's frontier
API prices — the number an engineering lead or a CFO actually cares about.

What this does
--------------
Lynx returns the relevant code in fewer tokens and fewer tool round-trips than
an agentic grep loop (see RESULTS.md). Those saved tokens are *input* tokens on
the model's next inference, so the saving converts directly to API spend at the
provider's input price. This script takes the measured per-query saving and a
team usage scenario and reports the monthly / yearly bill it removes — for the
flagship model of both major providers.

Two figures, both reported, so the claim is honest:

  - FLOOR (fully measured, zero assumptions): the saving is just the smaller
    tool output Lynx hands back per query (Django benchmark: 4,150 -> 1,725
    tokens to answer, -58%). No modelling of context size or round-trips.

  - WITH THE SAVED ROUND TRIP (one stated assumption): grep needs at least one
    extra tool round-trip to get the code into context (search, THEN read the
    file), and every round-trip re-sends the whole conversation. Removing one
    round-trip therefore also saves one context window of input tokens. That
    context is re-read from the prompt cache, which both providers bill at a
    fraction of the input price, so the saving is `--avg-context-tokens` x
    `--cache-read-discount` (defaults: 20,000 tokens at 10%, i.e. 2,000 tokens
    at full price). Those two knobs are the only modelled assumption;
    everything else stays measured.

Two config files, both editable, nothing else to touch:
  - `benchmarks/pricing.json`  — model prices ($/1M, input side, as of 2026-06).
  - `benchmarks/measured.json` — the per-query token deltas from each benchmarked
    codebase (Django/Python and Json.NET/C#). The numbers are computed every
    codebase × every model, so the report is honest about both — and leads with
    the bigger one.

The interactive page `savings_calculator.html` reads the same two files and lets
a user pick the codebase and the model from drop-downs and override the price live.

Reproduce
---------
    python benchmarks/savings_calculator.py                 # all codebases, all models
    python benchmarks/savings_calculator.py --devs 50 --queries-per-dev-day 80
    python benchmarks/savings_calculator.py --model "GPT-5.5" --input-price 4.5
    python benchmarks/savings_calculator.py --chart docs/img/cost_savings.svg

Both JSON files have embedded fallbacks, so the chart regenerates from a clean
checkout without re-running the long benchmark.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent

# Fallback prices if pricing.json is missing — keep in sync with that file.
FALLBACK_PRICES = [
    {"label": "Claude Fable 5",  "input": 10.0, "output": 50.0, "note": "Anthropic flagship", "chart": True},
    {"label": "Claude Opus 4.8", "input": 5.0,  "output": 25.0, "note": "top coding model",   "chart": True},
    {"label": "GPT-5.5",         "input": 5.0,  "output": 30.0, "note": "OpenAI flagship",     "chart": True},
]

# Published per-query token deltas — overridden by measured.json if present.
FALLBACK_CODEBASES = [
    {"label": "Django 5.2", "language": "Python", "lines": 157865, "grep": 4150, "lynx": 1725},
    {"label": "Json.NET",   "language": "C#",     "lines": 69132,  "grep": 6590, "lynx": 1540},
    {"label": "Guava",      "language": "Java",   "lines": 180517, "grep": 5892, "lynx": 807},
]


def load_prices() -> tuple[dict, list[str], str]:
    """Read pricing.json → (prices_by_label, chart_labels, as_of).

    prices_by_label maps each model's label to {"in", "out", "note"}.
    chart_labels is the subset flagged for the SVG chart (falls back to all).
    """
    f = BENCH_DIR / "pricing.json"
    if f.exists():
        spec = json.loads(f.read_text(encoding="utf-8"))
        models, as_of = spec["models"], spec.get("as_of", "")
    else:
        models, as_of = FALLBACK_PRICES, "2026-06"
    prices = {
        m["label"]: {"in": m["input"], "out": m["output"], "note": m.get("note", "")}
        for m in models
    }
    chart = [m["label"] for m in models if m.get("chart")] or list(prices)
    return prices, chart, as_of


def load_codebases() -> list[dict]:
    """Read measured.json → [{label, language, lines, grep, lynx}, ...].

    Each entry is one benchmarked codebase's *median tokens to answer*, for
    grep and for Lynx. Falls back to the published deltas if the file is absent.
    """
    f = BENCH_DIR / "measured.json"
    if f.exists():
        spec = json.loads(f.read_text(encoding="utf-8"))
        return [
            {"label": c["label"], "language": c["language"], "lines": c.get("lines"),
             "grep": c["grep_tokens"], "lynx": c["lynx_tokens"]}
            for c in spec["codebases"]
        ]
    return [dict(c) for c in FALLBACK_CODEBASES]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--devs", type=int, default=25,
                    help="engineers using AI coding assistants (default 25)")
    ap.add_argument("--queries-per-dev-day", type=int, default=60,
                    help="retrieval calls per engineer per working day (default 60)")
    ap.add_argument("--work-days-month", type=int, default=21,
                    help="working days per month (default 21)")
    ap.add_argument("--avg-context-tokens", type=int, default=20000,
                    help="avg live conversation size re-sent per round-trip, "
                         "for the second figure (default 20,000; set 0 for floor only)")
    ap.add_argument("--cache-read-discount", type=float, default=0.1,
                    help="fraction of the input price a cached context re-read costs "
                         "(default 0.1: both providers bill prompt-cache reads at about "
                         "a tenth; set 1.0 for no caching)")
    ap.add_argument("--model", default=None,
                    help="restrict the table to one model label from pricing.json")
    ap.add_argument("--input-price", type=float, default=None,
                    help="override the input $/1M of --model (the CLI analog of "
                         "editing the price field in the interactive page)")
    ap.add_argument("--chart", default=None, help="write an SVG bar chart here")
    args = ap.parse_args()

    prices, chart_labels, as_of = load_prices()
    if args.model:
        if args.model not in prices:
            ap.error(f"unknown --model {args.model!r}; choices: {', '.join(prices)}")
        if args.input_price is not None:
            prices[args.model]["in"] = args.input_price
        prices = {args.model: prices[args.model]}
        chart_labels = [args.model]
    elif args.input_price is not None:
        ap.error("--input-price requires --model")

    codebases = load_codebases()
    queries_month = args.devs * args.queries_per_dev_day * args.work_days_month
    # The re-read context counts at the cached price: 20k tokens at 10% is
    # worth 2k tokens at the full input price.
    ctx = int(round(args.avg_context_tokens * args.cache_read_discount))

    print(f"Team: {args.devs} devs x {args.queries_per_dev_day} retrievals/day "
          f"x {args.work_days_month} days = {queries_month:,} retrievals/month")

    for cb in codebases:
        floor_q = cb["grep"] - cb["lynx"]
        real_q = floor_q + ctx
        pct = 1 - cb["lynx"] / cb["grep"]
        tok_floor, tok_real = queries_month * floor_q, queries_month * real_q
        print(f"\n=== {cb['label']} ({cb['language']}): grep {cb['grep']:,} -> Lynx "
              f"{cb['lynx']:,} tokens to answer  ({pct:.0%} fewer)  |  saved/mo "
              f"floor {tok_floor/1e6:.1f}M  with round trip(+{ctx/1000:.1f}k cached) {tok_real/1e6:.1f}M ===")
        header = f"{'Model':18} {'$/1M in':>8} {'floor $/mo':>11} {'floor $/yr':>11} {'rt $/mo':>11} {'rt $/yr':>11}"
        print(header)
        print("-" * len(header))
        for name, p in prices.items():
            floor_mo, real_mo = tok_floor / 1e6 * p["in"], tok_real / 1e6 * p["in"]
            print(f"{name:18} {p['in']:>8.1f} {floor_mo:>11,.0f} {floor_mo*12:>11,.0f} "
                  f"{real_mo:>11,.0f} {real_mo*12:>11,.0f}")

    if args.chart:
        cl = [n for n in chart_labels if n in prices] or list(prices)
        write_chart(Path(args.chart), codebases, prices, cl, queries_month, args)
        print(f"\nchart: {args.chart}")
    return 0


# Per-codebase bar colours (cycled). Amber is the brand lead.
_CB_COLORS = ["#e8742c", "#2f81f7", "#1a7f5a", "#a371f7"]


def write_chart(out: Path, codebases, prices, chart_labels, queries_month, args) -> None:
    """Hand-rolled SVG (deterministic, no matplotlib): grouped bars, for each
    flagship model one bar per benchmarked codebase, showing the yearly API bill
    Lynx removes when the saved round trip is counted at the cached price. The
    floor is in the CLI table and in the footnote."""
    INK, SUB, GRID = "#24292f", "#57606a", "#e1e4e8"
    ctx = int(round(args.avg_context_tokens * args.cache_read_discount))
    ctx_full = args.avg_context_tokens
    disc = args.cache_read_discount
    models = chart_labels
    # realistic $/yr per (model, codebase)
    def yr(cb, price):
        return queries_month * (cb["grep"] - cb["lynx"] + ctx) / 1e6 * price * 12

    W, H = 900, 470
    pad_l, pad_r, pad_t, pad_b = 76, 30, 132, 78
    plot_w, plot_h = W - pad_l - pad_r, H - pad_t - pad_b
    maxv = max(yr(cb, prices[m]["in"]) for m in models for cb in codebases) * 1.16
    group_w = plot_w / len(models)
    bw = min(64, group_w / (len(codebases) + 1.2))

    def y(v):
        return pad_t + plot_h - plot_h * v / maxv

    bars, labels = [], []
    for i, m in enumerate(models):
        gx0 = pad_l + group_w * i + (group_w - bw * len(codebases)) / 2
        cx = pad_l + group_w * (i + 0.5)
        for j, cb in enumerate(codebases):
            v = yr(cb, prices[m]["in"])
            x = gx0 + j * bw
            col = _CB_COLORS[j % len(_CB_COLORS)]
            bars.append(
                f'<rect x="{x:.0f}" y="{y(v):.0f}" width="{bw-6:.0f}" '
                f'height="{pad_t+plot_h-y(v):.0f}" rx="3" fill="{col}"/>'
                f'<text x="{x+(bw-6)/2:.0f}" y="{y(v)-6:.0f}" font-size="12" '
                f'font-weight="800" fill="{col}" text-anchor="middle">${v/1000:,.0f}k</text>')
        labels.append(
            f'<text x="{cx:.0f}" y="{pad_t+plot_h+22:.0f}" font-size="14" '
            f'font-weight="700" fill="{INK}" text-anchor="middle">{m}</text>'
            f'<text x="{cx:.0f}" y="{pad_t+plot_h+40:.0f}" font-size="11" '
            f'fill="{SUB}" text-anchor="middle">{prices[m]["note"]} · ${prices[m]["in"]:.0f}/1M</text>')

    gridlines = []
    for frac in (0.25, 0.5, 0.75, 1.0):
        gv = maxv * frac
        gy = y(gv)
        gridlines.append(
            f'<line x1="{pad_l}" y1="{gy:.0f}" x2="{W-pad_r}" y2="{gy:.0f}" stroke="{GRID}"/>'
            f'<text x="{pad_l-8}" y="{gy+4:.0f}" font-size="11" fill="{SUB}" '
            f'text-anchor="end">${gv/1000:,.0f}k</text>')

    # Horizontal legend on its own row (colour → codebase + measured delta),
    # below the subtitle so nothing overlaps and nothing runs off the edge.
    legend = []
    lx = 36.0
    for j, cb in enumerate(codebases):
        col = _CB_COLORS[j % len(_CB_COLORS)]
        pct = 1 - cb["lynx"] / cb["grep"]
        txt = f'{cb["label"]} ({cb["language"]}) −{pct:.0%}'
        legend.append(
            f'<rect x="{lx:.0f}" y="78" width="13" height="13" rx="2" fill="{col}"/>'
            f'<text x="{lx+19:.0f}" y="89" font-size="12.5" fill="{INK}">{txt}</text>')
        lx += 19 + len(txt) * 7.0 + 26

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="-apple-system,'Segoe UI',Helvetica,Arial,sans-serif">
  <rect x="0" y="0" width="{W}" height="{H}" rx="12" fill="#ffffff" stroke="#d0d7de"/>
  <text x="36" y="38" font-size="21" font-weight="800" fill="{INK}">Yearly API bill Lynx removes, {args.devs} engineers</text>
  <text x="36" y="60" font-size="13" fill="{SUB}">Measured token delta + the saved round trip at the cached price · 2026-06 flagship input prices · {queries_month:,} retrievals/month</text>
  {''.join(legend)}
  {''.join(gridlines)}
  {''.join(bars)}
  {''.join(labels)}
  <text x="36" y="{H-18}" font-size="10.5" fill="{SUB}">Per retrieval: measured delta + one saved grep round trip re-reading a {ctx_full//1000}k context from the prompt cache at {disc:.0%} of the input price. Floor (tool output only) in benchmarks/RESULTS*.md.</text>
</svg>
'''
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(svg, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    raise SystemExit(main())
