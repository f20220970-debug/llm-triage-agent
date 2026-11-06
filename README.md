# Inbox Triage Agent (LLM + guardrails + daily eval)

[![daily-eval](https://github.com/f20220970-debug/llm-triage-agent/actions/workflows/daily-eval.yml/badge.svg)](https://github.com/f20220970-debug/llm-triage-agent/actions/workflows/daily-eval.yml)

**Live results page:** `https://f20220970-debug.github.io/llm-triage-agent/`, updated every day by GitHub Actions.

This is an AI agent that triages inbound messages for a US small-business insurance brokerage: quote, certificate, policy change, claim, billing or cancel. It's built the way a production agent should be: safety rules before the model, a fallback when the model fails, a router that decides what the agent may do alone, and an **evaluation that runs every day by itself** and publishes its results.

> **The messages are synthetic** (1,198 labelled examples), not customer data. It's built for Soma AI's Founder's Office application.

## How it works

```
message ─► 1. guardrail: claim words (accident, injury, fire, lawsuit...) ─► human, urgent  (no model call)
        └► 2. LLM classifier (Gemini, JSON output: intent, confidence, reason)
               └ no key / API error / bad JSON ─► local TF-IDF model (logged as a fallback)
        └► 3. router: auto-handle only certificate, billing or quote at confidence ≥ 0.85
                      claims ─► human, urgent · cancels and policy changes ─► human, always
```

- **Every day at 08:00 IST** (`.github/workflows/daily-eval.yml`), the workflow runs the tests and then the agent on 25 messages from each of two test sets (sized for the free tier):
  - **familiar wording**
  - **new wording:** phrasings the local model has never seen, which is how real inboxes drift
- **What it records:** accuracy, auto-handled rate, accuracy of auto-handled messages, the share of claims sent to a human, fallback rate and latency.
- **What it publishes:** it appends to `docs/history.csv`, rebuilds `docs/index.html` and commits both back to the repo.

## First run (local fallback only, no API key yet)

| | Familiar wording | New wording |
| --- | --- | --- |
| Accuracy | 97.5% | **20.0%** |
| Auto-handled | 52.5% | 15.0%, at **0% accuracy** |
| Claims sent to a human | 100% | 100% |

The local model breaks on new wording, and does so **confidently**, which is why the LLM layer and the daily new-wording eval exist. Once the Gemini key is added, the history table will show the LLM against this baseline, day by day.

The guardrail deliberately over-escalates. A certificate request that also mentions a slip-and-fall goes to a human. Missing a claim costs far more than an extra human review.

## Set up (5 minutes)

1. **Get a free Gemini key:** go to aistudio.google.com → Get API key.
2. **Add it to the repo:** in the repo, go to Settings → Secrets and variables → Actions → New repository secret. Name it `GEMINI_API_KEY`.
3. **Turn on the results page:** in the repo, go to Settings → Pages → Deploy from branch → `main` → folder `/docs`.
4. **Run it once:** go to Actions → daily-eval → Run workflow. After that it runs every day by itself.

To use a different model, add a repository *variable* `GEMINI_MODEL`. The default is `gemini-3.8-flash`.

## Run locally

```
pip install -r requirements.txt
pytest -q                                   # guardrail, JSON validation, API-failure fallback, routing, page build
GEMINI_API_KEY=... python agent/evaluate.py # or leave the key unset to run on the local fallback
```

## Files

```
agent/triage.py     guardrail, Gemini REST client (with retries), local fallback, router
agent/evaluate.py   daily sample -> scores -> docs/history.csv + docs/index.html
data/messages.jsonl 1,198 labelled synthetic messages, split into familiar / new wording
tests/              5 tests with a fake LLM, so CI never needs a key
```
