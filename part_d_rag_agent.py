import json, re, time, hashlib, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from mock_llm import chat
from common import POLICIES, CLEAN, OUTPUTS, STATE, write_jsonl

def chunk_text(text, size=40, overlap=10):
    words=text.split()
    step=size-overlap
    return [" ".join(words[i:i+size]) for i in range(0,len(words),step) if words[i:i+size]]

class PolicyRAG:
    def __init__(self, overlap=10):
        self.docs=[]
        for path in sorted(POLICIES.glob("*.txt")):
            for idx,chunk in enumerate(chunk_text(path.read_text(encoding="utf-8"),40,overlap)):
                self.docs.append({"doc":path.name,"chunk_id":idx,"text":chunk})
        self.vectorizer=TfidfVectorizer(stop_words="english")
        self.matrix=self.vectorizer.fit_transform([d["text"] for d in self.docs])
    def search(self, query, k=3):
        q=self.vectorizer.transform([query])
        scores=cosine_similarity(q,self.matrix)[0]
        order=scores.argsort()[::-1][:k]
        return [{**self.docs[i],"score":float(scores[i])} for i in order if scores[i]>0]
    def answer(self, question):
        hits=self.search(question)
        context="\n".join(f"[{h['doc']}] {h['text']}" for h in hits)
        msg=[{"role":"system","content":"POLICY ASSISTANT. Answer only from supplied policy context; if absent, say not found."},
             {"role":"user","content":f"CONTEXT:\n{context}\nQUESTION:\n{question}"}]
        response=chat(msg)
        return response.choices[0].message.content or "", hits

def evaluate_rag():
    qa=pd.read_csv(Path(__file__).resolve().parents[1]/"data/raw/policy_eval.csv").fillna("")
    rows=[]
    for overlap in (0,10):
        rag=PolicyRAG(overlap=overlap)
        for _,r in qa.iterrows():
            answer,hits=rag.answer(r.question)
            expected=r.expected_fact
            # NONE questions should be refused/not-found, not answered from unrelated context.
            correct=(("not find" in answer.lower() or "could not" in answer.lower())
                     if r.expected_doc=="NONE" else expected.lower() in answer.lower())
            rows.append({"overlap":overlap,"question":r.question,"expected_doc":r.expected_doc,
                         "expected_fact":expected,"answer":answer,"top_doc":hits[0]["doc"] if hits else "",
                         "top_score":hits[0]["score"] if hits else 0.0,"correct":bool(correct)})
    pd.DataFrame(rows).to_csv(OUTPUTS/"policy_eval_results.csv",index=False)
    return rows

class RefundAgent:
    def __init__(self):
        self.orders=pd.read_csv(CLEAN/"orders.csv").fillna("")
        self.rag=PolicyRAG(10)
        self.ledger=[]
        self.current_order=None
        self.last_calculation=None
    def get_order(self, order_id):
        found=self.orders[self.orders.order_id==order_id]
        if found.empty: return {"error":"order_not_found"}
        r=found.iloc[0]
        self.current_order=r.to_dict()
        self.last_calculation=None
        return {k:(float(r[k]) if k=="order_amount" else str(r[k])) for k in
                ["order_id","customer_id","status","order_amount","delivery_minutes","order_ts"]}
    def search_policy(self, query):
        hits=self.rag.search(query,3)
        return {"snippets":[{"doc":h["doc"],"text":h["text"]} for h in hits]}
    def calculate_refund(self, order_amount, delivery_minutes):
        try: delivery=float(delivery_minutes)
        except (ValueError,TypeError): delivery=-1
        amount=float(order_amount) if delivery>45 else 0.0
        # Synthetic scenario assumption: request is logged one day after the order timestamp.
        # This makes the seven-day policy window explicit and reproducible for this offline demo.
        within_window=False
        if self.current_order is not None:
            try:
                order_date=pd.to_datetime(self.current_order["order_ts"])
                simulated_request_date=order_date + pd.Timedelta(days=1)
                within_window=(simulated_request_date-order_date).days <= 7
            except Exception:
                within_window=False
        if not within_window: amount=0.0
        self.last_calculation={"eligible":amount>0,"amount":round(amount,2)}
        return {"refund_inr":round(amount,2),"eligible":amount>0,"within_7_day_window":within_window}
    def issue_refund(self, order_id, refund_inr):
        if not self.last_calculation or not self.last_calculation["eligible"] or float(refund_inr)>1000:
            return {"order_id":order_id,"refund_inr":float(refund_inr),"status":"denied_by_tool_guardrail"}
        record={"order_id":order_id,"refund_inr":float(refund_inr),"status":"issued_simulation_only"}
        self.ledger.append(record); return record
    def escalate_to_human(self, order_id, refund_inr):
        if float(refund_inr)<=1000:
            return {"order_id":order_id,"refund_inr":float(refund_inr),"status":"denied_invalid_escalation_amount"}
        record={"order_id":order_id,"refund_inr":float(refund_inr),"status":"human_approval_required"}
        self.ledger.append(record); return record
    def tool(self,name,args):
        tools={"get_order":self.get_order,"search_policy":self.search_policy,
               "calculate_refund":self.calculate_refund,"issue_refund":self.issue_refund,
               "escalate_to_human":self.escalate_to_human}
        if name not in tools: return {"error":"unknown_tool"}
        return tools[name](**args)
    def run(self, scenario, max_steps=6, max_seconds=8, max_tokens=4000):
        messages=[{"role":"system","content":"AGENT. Use tools to check order, policy and refund eligibility. Never issue a refund unless policy allows it."},
                  {"role":"user","content":scenario}]
        trace=[]; total_tokens=0; start=time.time(); final=""
        for step in range(1,max_steps+1):
            if time.time()-start>max_seconds: final="HANDOFF: time limit reached"; break
            try:
                response=chat(messages,tools=[])
            except Exception as exc:
                final=f"HANDOFF: tool/agent error ({type(exc).__name__})"
                break
            total_tokens+=response.usage.total_tokens
            if total_tokens>max_tokens: final="HANDOFF: token budget reached"; break
            msg=response.choices[0].message
            if not msg.tool_calls:
                final=msg.content or ""; break
            tc=msg.tool_calls[0]
            try:
                raw_args = tc.function.arguments if hasattr(tc,"function") else tc.arguments
                args = raw_args if isinstance(raw_args, dict) else json.loads(raw_args)
            except Exception: args={}
            name=tc.function.name if hasattr(tc,"function") else tc.name
            result=self.tool(name,args)
            trace.append({"step":step,"tool":name,"arguments":args,"result":result})
            messages.append({"role":"assistant","content":None,"tool_calls":[{"id":tc.id,"name":name,"arguments":args}]})
            messages.append({"role":"tool","name":name,"content":json.dumps(result),"tool_call_id":tc.id})
        else:
            final="HANDOFF: maximum six tool steps reached"
        if not final: final="HANDOFF: agent stopped without a final answer"
        return {"scenario":scenario,"final":final,"steps":trace,"total_tokens":total_tokens,
                "elapsed_seconds":round(time.time()-start,4),"handoff":final.startswith("HANDOFF")}

def main():
    eval_rows=evaluate_rag()
    orders=pd.read_csv(CLEAN/"orders.csv").fillna("")
    # Pick reproducible completed orders with different refund decision paths.
    completed=orders[orders.status=="completed"].copy()
    completed["delivery_num"]=pd.to_numeric(completed.delivery_minutes,errors="coerce")
    candidates=[]
    groups=[
        completed[(completed.delivery_num<=45) & (completed.order_amount.astype(float)<=1000)],
        completed[(completed.delivery_num>45) & (completed.order_amount.astype(float)<=1000)],
        completed[(completed.delivery_num>45) & (completed.order_amount.astype(float)>1000)],
    ]
    for subset in groups:
        subset=subset.sort_values("order_amount")
        if len(subset): candidates.append(subset.iloc[len(subset)//2])
    cancelled=orders[orders.status=="cancelled"]
    if len(cancelled): candidates.append(cancelled.iloc[0])
    agent=RefundAgent()
    scenarios=[f"Please check refund eligibility for order {r.order_id}." for r in candidates]
    scenarios += ["Please check refund eligibility for order ORD-999999.",
                  "Please check refund eligibility but I do not have the order identifier.",
                  "[LOOP] Please keep searching until you find a refund policy."]
    runs=[]; ledger=[]
    # Checkpoint: each completed scenario is written immediately; resume skips completed IDs.
    checkpoint=STATE/"agent_checkpoint.json"
    existing=set()
    prior_runs=[]
    prior_path=OUTPUTS/"agent_runs.jsonl"
    if prior_path.exists():
        for line in prior_path.read_text(encoding="utf-8").splitlines():
            try: prior_runs.append(json.loads(line))
            except json.JSONDecodeError: pass
    # Rebuild the simulated ledger from completed traces so resume does not lose earlier actions.
    for prior in prior_runs:
        for step in prior.get("steps",[]):
            result=step.get("result",{})
            if result.get("status") in {"issued_simulation_only","human_approval_required"}:
                agent.ledger.append(result)
    if checkpoint.exists():
        try: existing=set(json.loads(checkpoint.read_text()).get("completed",[]))
        except Exception: existing=set()
    for i,scenario in enumerate(scenarios):
        sid=f"scenario_{i+1}"
        if sid in existing: continue
        result=agent.run(scenario)
        result["scenario_id"]=sid
        runs.append(result)
        existing.add(sid)
        checkpoint.write_text(json.dumps({"completed":sorted(existing)},indent=2),encoding="utf-8")
        with (OUTPUTS/"agent_runs.jsonl").open("a",encoding="utf-8") as f:
            f.write(json.dumps(result,default=str)+"\n")
    # Plan-execute baseline: deterministic fixed plan, compared on same set of scenarios.
    plan=[]
    for s in scenarios:
        m=re.search(r"ORD-\d{6}",s)
        order=agent.get_order(m.group(0)) if m else {"error":"missing_order_id"}
        if "error" in order: answer="NOT_FOUND"
        elif order["status"]!="completed": answer="DENIED_CANCELLED"
        else:
            calc=agent.calculate_refund(order["order_amount"],order["delivery_minutes"])
            if calc["refund_inr"]<=0: answer="DENIED_NOT_LATE"
            elif calc["refund_inr"]>1000: answer="ESCALATE_HUMAN"
            else: answer="ELIGIBLE_REFUND"
        plan.append({"scenario":s,"plan_execute_result":answer})
    pd.DataFrame(plan).to_csv(OUTPUTS/"plan_execute_comparison.csv",index=False)
    # Persist only this run's simulated actions; agent execution never moves actual money.
    write_jsonl(OUTPUTS/"refund_ledger.jsonl",agent.ledger)
    (OUTPUTS/"part_d_summary.json").write_text(json.dumps({
        "rag_chunks_overlap_10":len(PolicyRAG(10).docs),
        "rag_eval_overlap_0_accuracy":sum(r["correct"] for r in eval_rows if r["overlap"]==0)/14,
        "rag_eval_overlap_10_accuracy":sum(r["correct"] for r in eval_rows if r["overlap"]==10)/14,
        "agent_scenarios":len(scenarios),"tool_step_limit":6,
        "request_date_assumption":"Synthetic support request is modeled as one day after order timestamp to test the 7-day policy guardrail.",
        "refund_ledger_note":"Simulation only; no payment or refund is actually executed.",
        "checkpoint_file":"state/agent_checkpoint.json"
    },indent=2),encoding="utf-8")
    print("Part D complete:",len(scenarios),"scenarios")

if __name__ == "__main__":
    main()
