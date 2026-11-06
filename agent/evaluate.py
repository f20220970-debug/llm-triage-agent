"""Daily evaluation run. Samples messages from both splits, runs the agent, scores it, appends to docs/history.csv
and rebuilds docs/index.html (served by GitHub Pages). Run by .github/workflows/daily-eval.yml every day."""
import csv, datetime as dt, html, json, os, random, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.triage import TriageAgent, GeminiClient, FallbackModel  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")


def load():
    with open(os.path.join(ROOT, "data", "messages.jsonl")) as fh:
        return [json.loads(l) for l in fh]


def score(rows):
    n = len(rows)
    if not n:
        return {}
    auto = [r for r in rows if r["route"] == "auto"]
    claims = [r for r in rows if r["intent"] == "claim_report"]
    return dict(n=n, accuracy=sum(r["pred"] == r["intent"] for r in rows) / n,
                auto_rate=len(auto) / n,
                auto_accuracy=(sum(r["pred"] == r["intent"] for r in auto) / len(auto)) if auto else None,
                claim_to_human=(sum(r["route"] == "human_urgent" for r in claims) / len(claims)) if claims else None,
                fallback_rate=sum(r["source"] == "fallback" for r in rows) / n,
                median_latency_ms=sorted(r["latency_ms"] for r in rows)[n // 2])


def run(sample_per_split=40, seed=None, llm=None, pause=0.0):
    msgs = load()
    train = [(m["text"], m["intent"]) for m in msgs if m["split"] == "familiar"]
    agent = TriageAgent(llm=llm if llm is not None else GeminiClient(), fallback=FallbackModel(train))
    rnd = random.Random(seed if seed is not None else dt.date.today().toordinal())
    out = {}
    detail = []
    for split in ["familiar", "new_wording"]:
        pool = [m for m in msgs if m["split"] == split]
        rows = []
        for i, m in enumerate(rnd.sample(pool, min(sample_per_split, len(pool))), 1):
            d = agent.classify(m["text"])
            if i % 5 == 0:
                print(f"{split}: {i}/{sample_per_split} done, last source={d.source}", flush=True)
            rows.append(dict(id=m["id"], split=split, text=m["text"], intent=m["intent"], pred=d.intent,
                             conf=round(d.confidence, 3), route=d.route, source=d.source, latency_ms=d.latency_ms))
            if pause and d.source == "llm":
                time.sleep(pause)                                   # stay inside free-tier rate limits
        out[split] = score(rows)
        detail += rows
    llm_used = sum(r["source"] == "llm" for r in detail)
    if not (agent.llm and agent.llm.available):
        mode = "fallback-only"
    elif llm_used == 0:
        mode = "llm-failed"
    else:
        mode = "llm"
    for e in agent.errors:
        print("LLM ERROR:", e)
    return dict(date=dt.date.today().isoformat(), mode=mode, model=getattr(agent.llm, "model", "none") if mode == "llm" else "none (local fallback)", **{
        f"{s}_{k}": v for s, d in out.items() for k, v in d.items()}), detail


def write_history(summary):
    os.makedirs(DOCS, exist_ok=True)
    path = os.path.join(DOCS, "history.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary))
        if new:
            w.writeheader()
        w.writerow(summary)
    with open(path) as fh:
        return list(csv.DictReader(fh))


def pct(x):
    return "n/a" if x in (None, "", "None") else f"{float(x) * 100:.1f}%"


def sparkline(values, w=320, h=60):
    vals = [float(v) for v in values if v not in (None, "", "None")]
    if len(vals) < 2:
        return "<em>trend appears after two daily runs</em>"
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1
    pts = " ".join(f"{i * w / (len(vals) - 1):.1f},{h - (v - lo) / span * (h - 8) - 4:.1f}" for i, v in enumerate(vals))
    return f'<svg width="{w}" height="{h}" role="img" aria-label="trend"><polyline fill="none" stroke="#1F3864" stroke-width="2" points="{pts}"/></svg>'


def write_page(hist, detail):
    last = hist[-1]
    errors = [r for r in detail if r["pred"] != r["intent"]][:15]
    rows_hist = "".join(
        f"<tr><td>{h['date']}</td><td>{h['mode']}</td><td>{pct(h.get('familiar_accuracy'))}</td><td>{pct(h.get('new_wording_accuracy'))}</td>"
        f"<td>{pct(h.get('new_wording_auto_accuracy'))}</td><td>{pct(h.get('familiar_auto_rate'))}</td><td>{pct(h.get('new_wording_claim_to_human'))}</td></tr>"
        for h in reversed(hist[-30:]))
    rows_err = "".join(f"<tr><td>{html.escape(r['text'])}</td><td>{r['intent']}</td><td>{r['pred']}</td><td>{r['conf']}</td><td>{r['source']}</td></tr>" for r in errors)
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Triage Agent Daily Eval</title>
<style>
:root{{--bg:#fff;--fg:#1a1a1a;--muted:#666;--line:#e5e5e5;--accent:#1F3864}}
@media (prefers-color-scheme:dark){{:root{{--bg:#111;--fg:#eee;--muted:#aaa;--line:#333;--accent:#8EA9DB}}}}
body{{background:var(--bg);color:var(--fg);font-family:system-ui,Arial,sans-serif;max-width:960px;margin:0 auto;padding:24px 16px}}
h1{{font-size:1.5rem}} .muted{{color:var(--muted)}} table{{border-collapse:collapse;width:100%;font-size:.9rem;margin:12px 0 28px}}
td,th{{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}} th{{color:var(--muted);font-weight:600}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:16px 0}}
.card{{border:1px solid var(--line);border-radius:8px;padding:12px}} .big{{font-size:1.6rem;font-weight:700;color:var(--accent)}}
.wrap{{overflow-x:auto}}
</style></head><body>
<h1>Inbox triage agent: daily evaluation</h1>
<p class="muted">Runs every day on GitHub Actions. Samples labelled messages (synthetic, not customer data), runs the agent
(claims guardrail, then an LLM classifier with a local fallback, then a confidence router) and scores it. Last run: {last['date']} · mode: {last['mode']} · model: {last['model']}</p>
<div class="cards">
<div class="card"><div class="muted">Accuracy, familiar wording</div><div class="big">{pct(last.get('familiar_accuracy'))}</div></div>
<div class="card"><div class="muted">Accuracy, new wording</div><div class="big">{pct(last.get('new_wording_accuracy'))}</div></div>
<div class="card"><div class="muted">Auto-handled (familiar)</div><div class="big">{pct(last.get('familiar_auto_rate'))}</div></div>
<div class="card"><div class="muted">Claims sent to a human</div><div class="big">{pct(last.get('new_wording_claim_to_human'))}</div></div>
</div>
<h2>New-wording accuracy over time</h2>{sparkline([h.get('new_wording_accuracy') for h in hist[-30:]])}
<h2>History (last 30 runs)</h2><div class="wrap"><table><tr><th>Date</th><th>Mode</th><th>Acc. familiar</th><th>Acc. new wording</th><th>Auto acc. (new)</th><th>Auto rate</th><th>Claims to human</th></tr>{rows_hist}</table></div>
<h2>Today's errors (up to 15)</h2><div class="wrap"><table><tr><th>Message</th><th>True</th><th>Predicted</th><th>Conf.</th><th>Source</th></tr>{rows_err}</table></div>
<p class="muted">Source: github.com/f20220970-debug/llm-triage-agent</p></body></html>"""
    with open(os.path.join(DOCS, "index.html"), "w") as fh:
        fh.write(page)


if __name__ == "__main__":
    pause = float(os.environ.get("LLM_PAUSE_SECONDS", "7"))
    n = int(os.environ.get("SAMPLE_PER_SPLIT", "25"))
    summary, detail = run(sample_per_split=n, pause=pause)
    print(json.dumps(summary, indent=1))
    if summary["mode"] == "llm-failed":
        sys.exit("Every Gemini call failed - see LLM ERROR lines above. Not publishing a misleading result.")
    hist = write_history(summary)
    write_page(hist, detail)
