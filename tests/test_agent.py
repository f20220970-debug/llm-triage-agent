import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.triage import TriageAgent, FallbackModel, parse_llm_json, guardrail
from agent import evaluate as E

msgs = E.load()
fb = FallbackModel([(m["text"], m["intent"]) for m in msgs if m["split"] == "familiar"])


class FakeLLM:
    """Stands in for Gemini so tests never need a key or network."""
    available, model = True, "fake"
    def __init__(self, reply): self.reply = reply
    def complete(self, prompt): return self.reply


class BrokenLLM(FakeLLM):
    def complete(self, prompt): raise TimeoutError("api down")


def test_guardrail_beats_everything():
    a = TriageAgent(llm=FakeLLM('{"intent":"billing","confidence":0.99}'), fallback=fb)
    d = a.classify("one of our trucks was in an accident")
    assert d.route == "human_urgent" and d.source == "guardrail"


def test_llm_json_is_validated():
    assert parse_llm_json('```json\n{"intent":"cancel","confidence":1.4}\n```')[:2] == ("cancel", 1.0)
    for bad in ['{"intent":"refund","confidence":0.9}', "sure! it is billing"]:
        try:
            parse_llm_json(bad); assert False
        except ValueError:
            pass


def test_api_failure_falls_back_to_local_model():
    a = TriageAgent(llm=BrokenLLM(""), fallback=fb)
    d = a.classify("please send a certificate of insurance to my client")
    assert d.source == "fallback" and a.fallback_count == 1


def test_low_confidence_or_risky_intent_goes_to_human():
    a = TriageAgent(llm=FakeLLM('{"intent":"cancel","confidence":0.99}'), fallback=fb)
    assert a.classify("I want to end things").route == "human"          # cancel is never auto
    a = TriageAgent(llm=FakeLLM('{"intent":"billing","confidence":0.5}'), fallback=fb)
    assert a.classify("when is my payment due").route == "human"       # below threshold


def test_daily_run_writes_page(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "DOCS", str(tmp_path))
    s, detail = E.run(sample_per_split=10, seed=1, llm=FakeLLM('{"intent":"new_quote","confidence":0.9}'))
    hist = E.write_history(s); E.write_page(hist, detail)
    assert (tmp_path / "index.html").exists() and s["familiar_n"] == 10
