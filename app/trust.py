from app.models import RetrievedChunk, HeatmapItem, FailureAnalysis

def classify_strength(score: float) -> str:
    if score >= 0.80:
        return "strong"
    elif score >= 0.60:
        return "medium"
    else:
        return "weak"

def build_heatmap(chunks: list[RetrievedChunk]) -> list[HeatmapItem]:
    heatmap = []
    for chunk in chunks:
        heatmap.append(HeatmapItem(
            chunk_id=chunk.chunk_id,
            score=chunk.score,
            strength=classify_strength(chunk.score)
        ))
    return heatmap

def suggest_reformulations(question: str) -> list[str]:
    q_lower = question.lower()
    reformulations = []
    
    if "summarize" in q_lower:
        reformulations.append(q_lower.replace("summarize", "overview of"))
        
    if "compare" in q_lower:
        reformulations.append(q_lower.replace("compare", "differences between"))
        
    if "all" in q_lower:
        reformulations.append(q_lower.replace("all", "main"))
        
    if not reformulations:
        words = [w for w in q_lower.split() if len(w) > 3]
        if words:
            key_term = " ".join(words[:2])
            reformulations.append(f"overview of '{key_term}'")
            reformulations.append(f"main aspects of '{key_term}'")
        else:
            reformulations.append(f"overview of {q_lower}")
            
    # Add quoted key terms if none of the above generated enough
    if len(reformulations) < 3:
        words = [w for w in q_lower.split() if len(w) > 3]
        if words:
            reformulations.append(f"details about '{words[0]}'")
            
    # Ensure uniqueness and return max 3
    return list(dict.fromkeys(reformulations))[:3]

def analyze_failure(question: str, confidence: float, evidence_coverage: float, chunks: list[RetrievedChunk]) -> FailureAnalysis:
    low_confidence = confidence < 0.55 or evidence_coverage < 0.50
    reasons = []
    
    if low_confidence:
        if not chunks:
            reasons.append("The document may not contain the answer.")
        else:
            if confidence < 0.40:
                reasons.append("The document may not contain the answer.")
                reasons.append("Retrieved chunks are only weakly related to the question.")
            else:
                reasons.append("The query may use different terminology than the document.")
                reasons.append("The answer may span multiple sections.")
                
        # Deduplicate reasons while preserving order
        unique_reasons = []
        for r in reasons:
            if r not in unique_reasons:
                unique_reasons.append(r)
        reasons = unique_reasons
            
    suggested_queries = suggest_reformulations(question) if low_confidence else []
    
    return FailureAnalysis(
        low_confidence=low_confidence,
        possible_reasons=reasons,
        suggested_queries=suggested_queries
    )
