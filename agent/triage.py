"""Inbox triage agent for an insurance brokerage.

Pipeline for each inbound message:
  1. Guardrail: claim keywords -> straight to a human, before any model is called (safety first, zero cost)
  2. Classifier: an LLM (Gemini, JSON output) returns intent + confidence + a one-line reason.
     If no API key is set, or the call fails, a local TF-IDF model is used instead and the fallback is logged.
  3. Router: auto-handle only low-risk intents above a confidence threshold; everything else goes to a human queue.
"""
import json, os, re, time, urllib.request, urllib.error
from dataclasses import dataclass, asdict

INTENTS = ["new_quote", "certificate_request", "policy_change", "claim_report", "billing", "cancel"]
AUTO_OK = {"certificate_request", "billing", "new_quote"}          # low-risk intents the agent may handle alone
CLAIM_WORDS = ["accident", "hurt", "injur", "slipped", "damage", "fire", "stole", "broke into", "suing", "sued",
               "rear-ended", "crash", "flood", "theft", "lawsuit"]

PROMPT = """You triage inbound messages for a US small-business insurance brokerage.
Classify the customer's PRIMARY intent as exactly one of:
new_quote, certificate_request, policy_change, claim_report, billing, cancel.
Definitions: certificate_request = proof of insurance / COI / certificate holder; policy_change = add/remove
vehicles, drivers, insureds, address or payroll; claim_report = any loss, injury, damage, theft or lawsuit;
billing = charges, invoices, payment schedule; cancel = end, stop or not renew coverage.
Return ONLY JSON: {{"intent": "...", "confidence": 0.0-1.0, "reason": "<12 words"}}
Message: {text}"""


@dataclass
class Decision:
    intent: str
    confidence: float
    route: str            # "auto" | "human" | "human_urgent"
    source: str           # "guardrail" | "llm" | "fallback"
    reason: str = ""
    latency_ms: int = 0


def guardrail(text):
    t = text.lower()
    return any(w in t for w in CLAIM_WORDS)


def parse_llm_json(raw):
    """Pull the first JSON object out of a model reply and validate it. Raises ValueError if unusable."""
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        raise ValueError("no JSON in reply")
    obj = json.loads(m.group(0))
    intent = str(obj.get("intent", "")).strip()
    if intent not in INTENTS:
        raise ValueError(f"unknown intent {intent!r}")
    conf = float(obj.get("confidence", 0))
    return intent, max(0.0, min(1.0, conf)), str(obj.get("reason", ""))[:120]


class GeminiClient:
    """Minimal REST client (no SDK) so the agent runs anywhere with the standard library."""

    def __init__(self, api_key=None, model=None, timeout=None):
        self.key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
        self.timeout = timeout or float(os.environ.get("LLM_TIMEOUT_SECONDS", "90"))   # newer "thinking" models are slow

    def _gen_config(self):
        cfg = {"temperature": 0, "responseMimeType": "application/json"}
        level = os.environ.get("GEMINI_THINKING_LEVEL", "")         # e.g. "low" to make thinking models answer faster
        if level:
            cfg["thinkingConfig"] = {"thinkingLevel": level}
        return cfg

    @property
    def available(self):
        return bool(self.key)

    def complete(self, prompt):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        body = json.dumps({"contents": [{"parts": [{"text": prompt}]}],
                           "generationConfig": self._gen_config()}).encode()
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json",
                                                              "x-goog-api-key": self.key})   # header works for AIza... and AQ... keys
        for attempt in range(3):                                     # retry on rate limits / transient errors
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    data = json.load(r)
                return data["candidates"][0]["content"]["parts"][0]["text"]
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 503) and attempt < 2:
                    retry = e.headers.get("Retry-After")
                    time.sleep(float(retry) if retry and retry.replace(".", "").isdigit() else 15 * (attempt + 1))
                    continue
                detail = e.read().decode("utf-8", "replace")[:300]
                raise RuntimeError(f"Gemini HTTP {e.code}: {detail}") from None


class FallbackModel:
    """Local TF-IDF + logistic regression, trained on the familiar-wording messages only."""

    def __init__(self, train):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline, make_union
        self.m = make_pipeline(make_union(TfidfVectorizer(ngram_range=(1, 2), min_df=2),
                                          TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2)),
                               LogisticRegression(max_iter=3000, C=4))
        self.m.fit([t for t, _ in train], [y for _, y in train])

    def predict(self, text):
        p = self.m.predict_proba([text])[0]
        i = p.argmax()
        return self.m.classes_[i], float(p[i])


class TriageAgent:
    def __init__(self, llm=None, fallback=None, threshold=0.85):
        self.llm, self.fallback, self.threshold = llm, fallback, threshold
        self.fallback_count = 0
        self.errors = []                                               # first few LLM errors, for the run log

    def classify(self, text):
        t0 = time.time()
        if guardrail(text):
            return Decision("claim_report", 1.0, "human_urgent", "guardrail", "claim keyword", int((time.time() - t0) * 1000))
        source, reason = "llm", ""
        try:
            if not (self.llm and self.llm.available):
                raise RuntimeError("no LLM configured")
            intent, conf, reason = parse_llm_json(self.llm.complete(PROMPT.format(text=text)))
        except Exception as e:                                       # API down, bad JSON, no key -> local model
            self.fallback_count += 1
            if len(self.errors) < 3 and self.llm and self.llm.available:
                self.errors.append(f"{type(e).__name__}: {e}")
                print(f"LLM ERROR (live): {type(e).__name__}: {str(e)[:300]}", flush=True)
            source, reason = "fallback", f"fallback: {type(e).__name__}"
            intent, conf = self.fallback.predict(text)
        if intent == "claim_report":
            route = "human_urgent"
        elif intent in AUTO_OK and conf >= self.threshold:
            route = "auto"
        else:
            route = "human"
        return Decision(intent, conf, route, source, reason, int((time.time() - t0) * 1000))
