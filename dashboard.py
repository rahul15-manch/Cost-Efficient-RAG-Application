import streamlit as st
import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Trust-Aware RAG", page_icon="🔍")

API_URL = "http://127.0.0.1:8000"

st.title("🔍 Trust-Aware Cost-Efficient RAG")

tab_ask, tab_trust, tab_bench, tab_export = st.tabs(["Ask", "Trust", "Benchmarks", "Exports"])

if "last_pipeline_res" not in st.session_state:
    st.session_state.last_pipeline_res = None

with tab_ask:
    st.header("Ask a Question")
    question = st.text_input("Question", value="What are the main components of the trust layer?")
    if st.button("Run RAG"):
        if question:
            with st.spinner("Running pipeline..."):
                try:
                    res = requests.post(f"{API_URL}/pipeline", json={"question": question})
                    if res.status_code == 200:
                        st.session_state.last_pipeline_res = res.json()
                    else:
                        st.error(f"API Error: {res.text}")
                except Exception as e:
                    st.error(f"Connection Error: {e}")

    if st.session_state.last_pipeline_res:
        data = st.session_state.last_pipeline_res
        
        conf = data["confidence"]
        cov = data["evidence_coverage"]
        top_k = len(data.get("citations", []))
        
        # Determine Trust Badge
        if conf >= 0.75:
            trust_badge = "🟢 High Trust"
        elif conf >= 0.45:
            trust_badge = "🟡 Medium Trust"
        else:
            trust_badge = "🔴 Low Trust"
            
        st.subheader(trust_badge)
        st.caption(f"Confidence {conf:.2f} • Coverage {cov:.2f} • top-k {top_k}")
        
        st.subheader("Grounded answer")
        st.write(data["answer"])
        
        st.subheader("Performance Metrics")
        metrics = data.get("metrics", {})
        col1, col2, col3 = st.columns(3)
        col1.metric("Retrieval (ms)", metrics.get("retrieval_ms", 0))
        col2.metric("Generation (ms)", metrics.get("generation_ms", 0))
        col3.metric("Total (ms)", metrics.get("total_ms", 0))
        
        st.subheader("Token Usage")
        tok = data.get("token_usage", {})
        st.write(f"Prompt: {tok.get('prompt_tokens', 0)} | Completion: {tok.get('completion_tokens', 0)}")
        
        st.subheader("Citations")
        for i, cit in enumerate(data.get("citations", [])):
            with st.expander(f"{cit['source']} — page {cit.get('page', 'N/A')}"):
                st.write(f"**Chunk ID:** {cit['chunk_id']}")
                st.write(f"**Section:** {cit.get('section', 'N/A')}")
                st.write(f"**Score:** {cit['score']}")
                if cit.get('text'):
                    st.write(cit['text'])

with tab_trust:
    st.header("Trust Diagnostics")
    if st.session_state.last_pipeline_res:
        data = st.session_state.last_pipeline_res
        conf = data["confidence"]
        cov = data["evidence_coverage"]
        
        if conf >= 0.75:
            st.success("High Trust (>=0.75)")
        elif conf >= 0.45:
            st.warning("Medium Trust (0.45-0.74)")
        else:
            st.error("Low Trust (<0.45)")
            
        st.progress(conf, text=f"Confidence: {conf:.2f}")
        st.progress(cov, text=f"Evidence Coverage: {cov:.2f}")
        
        if conf < 0.35 or cov < 0.30:
            st.error("Abstention triggered: insufficient evidence.")
    else:
        st.info("Run a query in the Ask tab first.")

with tab_bench:
    st.header("Benchmark Comparison")
    if os.path.exists("eval/results.json"):
        with open("eval/results.json") as f:
            summary = json.load(f)
            
        ret = summary.get("retrieval", {})
        gen = summary.get("generation", {})
        
        st.subheader("Fixed-k vs Adaptive-k")
        if os.path.exists("eval/results_retrieval.csv"):
            df_ret = pd.read_csv("eval/results_retrieval.csv")
            st.dataframe(df_ret)
            
        st.subheader("Charts")
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Latency chart
        axes[0].bar(["Fixed-k", "Adaptive-k"], [ret.get("latency_fixed", 0), ret.get("latency_adaptive", 0)], color=['#ff9999', '#66b3ff'])
        axes[0].set_title("Latency (ms)")
        axes[0].set_ylabel("ms")
        
        # Token chart 
        axes[1].bar(["Avg Tokens"], [gen.get("tokens", {}).get("mean", 0)], color=['#99ff99'])
        axes[1].set_title("Token Usage")
        axes[1].set_ylabel("Tokens")
        
        # Retrieval quality chart
        f_rec = ret.get("fixed", {}).get("recall@5", 0)
        a_rec = ret.get("adaptive", {}).get("recall@5", 0)
        axes[2].bar(["Fixed-k", "Adaptive-k"], [f_rec, a_rec], color=['#ffcc99', '#c2c2f0'])
        axes[2].set_title("Recall@5")
        axes[2].set_ylim([0, 1.05])
        
        st.pyplot(fig)
    else:
        st.info("No benchmark results found. Run evaluation first.")

with tab_export:
    st.header("Export Reports")
    
    for fname in ["results.json", "results.csv", "benchmark_report.md"]:
        path = f"eval/{fname}"
        if os.path.exists(path):
            with open(path, "r") as f:
                data = f.read()
            st.download_button(
                label=f"Download {fname}",
                data=data,
                file_name=fname,
                mime="text/plain"
            )
        else:
            st.write(f"File not found: {fname}")
