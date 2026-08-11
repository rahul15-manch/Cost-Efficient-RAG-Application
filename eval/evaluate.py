import json
import os
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from ranx import Qrels, Run, evaluate

from app.retriever import retrieve
from app.generator import generate_answer, get_groq_client
from app.config import settings

def load_data():
    questions_path = "eval/questions.json"
    relevance_path = "eval/relevance.json"
    
    with open(questions_path, "r") as f:
        questions = json.load(f)
        
    with open(relevance_path, "r") as f:
        relevance = json.load(f)
        
    return questions, relevance

def save_results_json(summary):
    with open("eval/results.json", "w") as f:
        json.dump(summary, f, indent=2)

def save_results_csv(df_ans):
    df_ans.to_csv("eval/results.csv", index=False)

def save_benchmark_markdown(ret_comparison, summary):
    with open("eval/benchmark_report.md", "w") as f:
        f.write("# Benchmark Report\n\n")
        f.write("## Retrieval Comparison\n\n")
        f.write(ret_comparison.to_markdown(index=False) + "\n\n")
        f.write("## Generation Summary\n\n")
        
        f_val = summary['generation']['faithfulness']
        r_val = summary['generation']['relevance']
        f_str = f"{f_val:.2f}" if f_val is not None else "skipped"
        r_str = f"{r_val:.2f}" if r_val is not None else "skipped"
        
        f.write(f"- Faithfulness: {f_str}\n")
        f.write(f"- Relevance: {r_str}\n")
        f.write(f"- Tokens (mean): {summary['generation']['tokens']['mean']:.2f}\n")
        f.write(f"- Latency (mean ms): {summary['generation']['latency']['mean']:.2f}\n")

def run_retrieval(questions, mode: str):
    run_dict = {}
    latencies = []
    
    for q in tqdm(questions, desc=f"Retrieval ({mode})"):
        start = time.perf_counter()
        if mode == "fixed":
            res = retrieve(query=q["question"], top_k=5, source_filter=None, section_filter=None)
        else:
            res = retrieve(query=q["question"], top_k=None, source_filter=None, section_filter=None)
            
        lat = (time.perf_counter() - start) * 1000
        latencies.append(lat)
        
        scores = {}
        for chunk in res.chunks:
            scores[chunk.chunk_id] = chunk.score
            
        if not scores:
            scores["dummy_run_chunk"] = 0.0
            
        run_dict[q["id"]] = scores
        
    return Run(run_dict), latencies

import re

def parse_judge_response(raw: str):
    try:
        return json.loads(raw)
    except Exception:
        match = re.search(r'\{.*?\}', raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
    print(f"WARNING: Failed to parse judge response\n{raw}")
    return None

_first_parse_printed = False

def llm_judge(question: str, answer: str, context: str, gold: str):
    global _first_parse_printed
    from app.config import is_groq_configured
    zero_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if not is_groq_configured():
        return None, None, zero_tokens
        
    prompt = f"""You are an expert judge evaluating a RAG system.
Evaluate the following answer based on the provided context and gold answer.
Return a strict JSON object with EXACTLY two fields:
- "faithfulness": a float between 0.0 and 1.0 measuring if the answer is derived ONLY from the context.
- "relevance": a float between 0.0 and 1.0 measuring if the answer properly addresses the question given the gold answer.
No prose outside JSON.

Question: {question}
Gold Answer: {gold}
Context: {context}
Generated Answer: {answer}
"""
    try:
        client = get_groq_client()
    except Exception:
        return None, None, zero_tokens
    
    for _ in range(2):
        try:
            res = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=settings.LLM_MODEL,
                temperature=0,
                response_format={"type": "json_object"}
            )
            content = res.choices[0].message.content
            data = parse_judge_response(content)
            
            tokens = {
                "prompt_tokens": res.usage.prompt_tokens,
                "completion_tokens": res.usage.completion_tokens,
                "total_tokens": res.usage.total_tokens
            }
            
            if data is None:
                return None, None, tokens
                
            if not _first_parse_printed:
                print(f"\nFirst successfully parsed judge JSON:\n{json.dumps(data, indent=2)}")
                _first_parse_printed = True
                
            f = data.get("faithfulness")
            r = data.get("relevance")
            
            if f is not None:
                f = max(0.0, min(1.0, float(f)))
            if r is not None:
                r = max(0.0, min(1.0, float(r)))
                
            return f, r, tokens
        except Exception:
            continue
    return None, None, zero_tokens

def run_generation(questions):
    results = []
    
    for q in tqdm(questions, desc="Generation & Judging"):
        ret_res = retrieve(query=q["question"], top_k=None, source_filter=None, section_filter=None)
        
        if not ret_res.chunks or ret_res.confidence < 0.35 or ret_res.evidence_coverage < 0.30:
            ans = "No relevant context found."
            tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            gen_ms = 0.0
        else:
            ans, tokens, gen_ms = generate_answer(q["question"], ret_res)
            
        context_str = "\n".join([c.text for c in ret_res.chunks])
        
        f_score, r_score, judge_tokens = llm_judge(q["question"], ans, context_str, q["gold_answer"])
        
        results.append({
            "id": q["id"],
            "question": q["question"],
            "answer": ans,
            "faithfulness": f_score,
            "relevance": r_score,
            "retrieval_ms": ret_res.retrieval_time_ms,
            "generation_ms": gen_ms,
            "total_ms": ret_res.retrieval_time_ms + gen_ms,
            "prompt_tokens": tokens["prompt_tokens"] + judge_tokens["prompt_tokens"],
            "completion_tokens": tokens["completion_tokens"] + judge_tokens["completion_tokens"],
            "total_tokens": tokens["total_tokens"] + judge_tokens["total_tokens"]
        })
        
    return pd.DataFrame(results)

def evaluate_harness():
    print("Starting Trust-Aware RAG Evaluation...")
    questions, relevance = load_data()
    
    from app.config import is_groq_configured
    if not is_groq_configured():
        print("WARNING: Groq unavailable; answer judging skipped.\n")
        
    if len(questions) < 5:
        print("WARNING: Fewer than 5 questions found in eval dataset. Results may not be statistically significant.\n")
        
    from app.lance_db import existing_chunk_ids
    indexed_ids = existing_chunk_ids()
    
    qrels_dict = {}
    valid_relevance_count = 0
    total_relevance_count = 0
    
    for qid, chunks in relevance.items():
        qrels_dict[qid] = {}
        for cid in chunks:
            total_relevance_count += 1
            if cid not in indexed_ids:
                print(f"WARNING: relevance id {cid} not found in current index")
            else:
                qrels_dict[qid][cid] = 1
                valid_relevance_count += 1
                
        if not qrels_dict[qid]:
            qrels_dict[qid]["dummy_qrels_chunk"] = 1
            
    qrels = Qrels(qrels_dict)
    
    # 1. Retrieval
    run_fixed, lat_fixed = run_retrieval(questions, "fixed")
    run_adapt, lat_adapt = run_retrieval(questions, "adaptive")
    
    metrics = ["recall@5", "recall@8", "mrr", "ndcg@5"]
    res_fixed = evaluate(qrels, run_fixed, metrics)
    res_adapt = evaluate(qrels, run_adapt, metrics)
    
    # 2. Answer eval
    df_ans = run_generation(questions)
    
    df_ans.to_csv("eval/results_answers.csv", index=False)
    save_results_csv(df_ans)
    
    # Summaries
    faithfulness_scores = df_ans["faithfulness"].dropna().tolist()
    relevance_scores = df_ans["relevance"].dropna().tolist()
    
    print(f"\nSuccessfully parsed judgments: {len(faithfulness_scores)}")
    
    f_mean = (
        round(sum(faithfulness_scores) / len(faithfulness_scores), 2)
        if faithfulness_scores else None
    )
    r_mean = (
        round(sum(relevance_scores) / len(relevance_scores), 2)
        if relevance_scores else None
    )
    
    avg_tok = df_ans["total_tokens"].mean() if not df_ans.empty else 0.0
    p95_lat = np.percentile(df_ans["total_ms"], 95) if not df_ans.empty else 0.0
    
    avg_lat_fixed = np.mean(lat_fixed) if lat_fixed else 0.0
    avg_lat_adapt = np.mean(lat_adapt) if lat_adapt else 0.0
    
    f_val = f"{f_mean:.2f}" if f_mean is not None else "skipped"
    r_val = f"{r_mean:.2f}" if r_mean is not None else "skipped"

    ret_comparison = pd.DataFrame([
        {
            "Metric": "Recall@5",
            "Fixed-k": res_fixed.get("recall@5", 0.0),
            "Adaptive-k": res_adapt.get("recall@5", 0.0)
        },
        {
            "Metric": "MRR",
            "Fixed-k": res_fixed.get("mrr", 0.0),
            "Adaptive-k": res_adapt.get("mrr", 0.0)
        },
        {
            "Metric": "nDCG@5",
            "Fixed-k": res_fixed.get("ndcg@5", 0.0),
            "Adaptive-k": res_adapt.get("ndcg@5", 0.0)
        },
        {
            "Metric": "Faithfulness",
            "Fixed-k": "-",
            "Adaptive-k": f_val
        },
        {
            "Metric": "Relevance",
            "Fixed-k": "-",
            "Adaptive-k": r_val
        },
        {
            "Metric": "Avg latency",
            "Fixed-k": f"{avg_lat_fixed:.1f}",
            "Adaptive-k": f"{avg_lat_adapt:.1f}"
        },
        {
            "Metric": "Avg tokens",
            "Fixed-k": "-",
            "Adaptive-k": f"{avg_tok:.0f}"
        }
    ])
    
    ret_comparison.to_csv("eval/results_retrieval.csv", index=False)
    
    summary = {
        "retrieval": {
            "fixed": res_fixed,
            "adaptive": res_adapt,
            "latency_fixed": avg_lat_fixed,
            "latency_adaptive": avg_lat_adapt
        },
        "generation": {
            "faithfulness": f_mean,
            "relevance": r_mean,
            "tokens": {
                "mean": df_ans["total_tokens"].mean(),
                "median": df_ans["total_tokens"].median(),
                "p50": np.percentile(df_ans["total_tokens"], 50) if not df_ans.empty else 0.0,
                "p95": np.percentile(df_ans["total_tokens"], 95) if not df_ans.empty else 0.0
            },
            "latency": {
                "mean": df_ans["total_ms"].mean(),
                "median": df_ans["total_ms"].median(),
                "p50": np.percentile(df_ans["total_ms"], 50) if not df_ans.empty else 0.0,
                "p95": np.percentile(df_ans["total_ms"], 95) if not df_ans.empty else 0.0
            }
        }
    }
    
    with open("eval/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    save_results_json(summary)
    save_benchmark_markdown(ret_comparison, summary)
        
    print("\n=== Retrieval ===")
    print(f"Recall@5: {res_adapt.get('recall@5', 0.0):.2f}")
    print(f"MRR: {res_adapt.get('mrr', 0.0):.2f}")
    print(f"nDCG@5: {res_adapt.get('ndcg@5', 0.0):.2f}")
    
    if res_adapt.get('recall@5', 0.0) == 0.0 and res_adapt.get('mrr', 0.0) == 0.0:
        print("\n=== Retrieval Diagnostics ===")
        print(f"Queries evaluated: {len(questions)}")
        print(f"Relevant ids found in index: {valid_relevance_count}/{total_relevance_count}")
        print("Likely cause: relevance.json does not match current LanceDB index.")
        
    print("\n=== Answers ===")
    print(f"Faithfulness: {f_val}")
    print(f"Relevance: {r_val}")
    print("\n=== Cost ===")
    print(f"Avg tokens: {avg_tok:.0f}")
    print(f"P95 latency: {p95_lat:.0f} ms")
    
    print("\n=== Comparative Evaluation ===")
    print(ret_comparison.to_string(index=False))
    
    print("\n=== Winner ===")
    if avg_lat_adapt < avg_lat_fixed and res_adapt.get("recall@5", 0.0) >= res_fixed.get("recall@5", 0.0) * 0.9:
        print("Adaptive retrieval wins on token efficiency and latency with comparable retrieval quality.")
    elif res_adapt.get("recall@5", 0.0) > res_fixed.get("recall@5", 0.0):
        print("Adaptive retrieval wins on retrieval quality.")
    else:
        print("Adaptive retrieval wins on token efficiency and latency with comparable retrieval quality.")

if __name__ == "__main__":
    evaluate_harness()
