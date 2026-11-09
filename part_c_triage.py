import json, re, time, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from mock_llm import chat, reset, TransientLLMError, est_tokens
from common import RAW, CLEAN, OUTPUTS

ALLOWED = {"billing","delivery","technical","fraud"}
EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE = re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{9}(?!\d)")
ORDER = re.compile(r"\bORD-\d{6}\b", re.I)

def redact(text):
    text = EMAIL.sub("[EMAIL_REDACTED]", str(text or ""))
    text = PHONE.sub("[PHONE_REDACTED]", text)
    return text

def classify_ticket(text, few_shot=False, max_retries=3):
    if not text or not str(text).strip():
        return {"category":"human_review","reason":"empty_ticket","attempts":0,"tokens":0}
    if len(text) > 1800:
        return {"category":"human_review","reason":"ticket_too_long","attempts":0,"tokens":0}
    safe = redact(text)
    system = "CLASSIFY customer support ticket. Return JSON with category and reason. Allowed: billing, delivery, technical, fraud."
    if few_shot:
        system += "\nEXAMPLES: fraud=unauthorized; technical=app crash; billing=double charge; delivery=late parcel."
    messages=[{"role":"system","content":system},{"role":"user","content":safe}]
    total_tokens=0
    for attempt in range(1,max_retries+1):
        try:
            response=chat(messages)
            total_tokens += response.usage.total_tokens
            content=response.choices[0].message.content or ""
            try:
                obj=json.loads(content)
                cat=obj.get("category")
                if cat not in ALLOWED or not isinstance(obj.get("reason"),str):
                    return {"category":"human_review","reason":"invalid_json_schema_or_category",
                            "attempts":attempt,"tokens":total_tokens}
                return {"category":cat,"reason":obj["reason"],"attempts":attempt,"tokens":total_tokens}
            except (json.JSONDecodeError,AttributeError):
                return {"category":"human_review","reason":"invalid_json_response",
                        "attempts":attempt,"tokens":total_tokens}
        except TransientLLMError:
            if attempt == max_retries:
                return {"category":"human_review","reason":"retry_exhausted",
                        "attempts":attempt,"tokens":total_tokens}
            time.sleep(0.01 * attempt)
    return {"category":"human_review","reason":"unhandled_error","attempts":max_retries,"tokens":total_tokens}

def main():
    tickets=pd.read_csv(RAW/"tickets.csv").fillna("")
    seg=pd.read_csv(OUTPUTS/"customer_segments.csv")[["customer_id","segment"]]
    tickets=tickets.merge(seg,on="customer_id",how="left")
    reset()
    rows=[]
    for _,r in tickets.iterrows():
        result=classify_ticket(r["text"],few_shot=True)
        category=result["category"]
        route = "security_team" if category=="fraud" else (
            "priority_support" if r.get("segment")=="Premium" else
            "human_review" if category=="human_review" else "standard_support")
        rows.append({"ticket_id":r.ticket_id,"customer_id":r.customer_id,
                     "predicted_category":category,"true_category":r.true_category,
                     "reason":result["reason"],"route":route,
                     "pii_redacted":True,"attempts":result["attempts"],"tokens":result["tokens"]})
    out=pd.DataFrame(rows)
    out.to_csv(OUTPUTS/"triage_results.csv",index=False)
    # Budget demonstration: compare zero-shot and few-shot token estimates on a fixed sample.
    demo=[]
    reset()
    for _,r in tickets.head(20).iterrows():
        safe=redact(r["text"])
        z="CLASSIFY ticket. Return JSON with allowed category and reason."
        f=z+"\nEXAMPLES: fraud=unauthorized; technical=app crash; billing=double charge; delivery=late parcel."
        zero=classify_ticket(r["text"],few_shot=False)
        few=classify_ticket(r["text"],few_shot=True)
        demo.append({"ticket_id":r.ticket_id,"true_category":r.true_category,
                     "zero_shot_prediction":zero["category"],"few_shot_prediction":few["category"],
                     "zero_shot_prompt_tokens":est_tokens(z)+est_tokens(safe),
                     "few_shot_prompt_tokens":est_tokens(f)+est_tokens(safe),
                     "budget_tokens":500,"few_shot_within_budget":est_tokens(f)+est_tokens(safe)<=500})
    pd.DataFrame(demo).to_csv(OUTPUTS/"triage_budget_demo.csv",index=False)
    (OUTPUTS/"part_c_summary.json").write_text(json.dumps({
        "tickets_processed":len(out),"human_review":int((out.predicted_category=="human_review").sum()),
        "fraud_routes":int((out.route=="security_team").sum()),
        "pii_redaction":"email and Indian mobile number patterns redacted before mock LLM call",
        "max_ticket_chars":1800,"max_retries":3
    },indent=2),encoding="utf-8")
    print("Part C complete:",len(out),"tickets")

if __name__ == "__main__":
    main()
