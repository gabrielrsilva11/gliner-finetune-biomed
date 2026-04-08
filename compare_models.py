"""
Builds knowledge graphs from case reports using two GLiNER models (baseline
and LoRA fine-tuned) via RetriCo, queries both with the same questions,
and scores answers using an LLM as judge.
"""

import json
import os
import requests
import retrico
from graphrag.document_ingestion import extract_texts_for_pipeline

BASELINE_MODEL = "knowledgator/gliner-relex-large-v1.0"
LORA_MODEL     = "grsilva/gliner-relex-biomedical-lora"
DATA_FILE      = "data/rag_documents/general_case_reports.json"
QUESTIONS_FILE = "test_questions.txt"
OUTPUT_FILE    = "data/eval_results/model_comparison.json"

API_KEY    = os.environ.get("DEEPSEEK_API_KEY", "")
LLM_MODEL  = "deepseek-chat"
BASE_URL   = "https://api.deepseek.com"

ENTITY_LABELS = [
    "PATIENT DEMOGRAPHIC",
    "DISEASE OR CONDITION",
    "SYMPTOM OR CLINICAL_FINDING",
    "DRUG OR MEDICATION",
    "PROCEDURE OR SURGERY",
    "DIAGNOSTIC TEST OR DEVICE",
    "ANATOMY",
    "GENETIC MUTATION OR MARKER",
]

RELATION_LABELS = [
    "TREATS OR MANAGES",
    "CAUSES OR ADVERSE EFFECT OF",
    "PRESENTS WITH",
    "DIAGNOSED BY",
    "MIMICS OR DIFFERENTIAL DIAGNOSIS",
    "LOCATED IN OR AFFECTS",
    "ASSOCIATED WITH GENETIC MARKER",
    "COMPLICATES OR CO-OCCURS WITH",
]

MODEL_CONFIGS = {
    "baseline": {
        "model": BASELINE_MODEL,
        "pipeline_name": "compare_baseline",
        "json_output": "data/graph_data/compare_baseline.json",
        "threshold": 0.4,
        "relation_threshold": 0.5,
    },
    "lora": {
        "model": LORA_MODEL,
        "pipeline_name": "compare_lora",
        "json_output": "data/graph_data/compare_lora.json",
        "threshold": 0.4,
        "relation_threshold": 0.5,
    },
}


def extract_retrieval_metrics(result) -> dict:
    """Extract entity, relation, and chunk counts from a RetriCo PipeContext result."""
    metrics = {
        "query_entities": 0,
        "retrieved_entities": 0,
        "retrieved_relations": 0,
        "retrieved_chunks": 0,
        "subgraph_density": 0.0,
        "entities": [],
        "relations": [],
        "chunks": [],
    }
    parser_output = result.get("parser_result")
    if parser_output and "entities" in parser_output:
        metrics["query_entities"] = len(parser_output["entities"])

    chunk_output = result.get("chunk_result")
    if chunk_output and "subgraph" in chunk_output:
        subgraph = chunk_output["subgraph"]
        e = len(subgraph.entities)
        r = len(subgraph.relations)
        metrics["retrieved_entities"] = e
        metrics["retrieved_relations"] = r
        metrics["retrieved_chunks"] = len(subgraph.chunks)
        metrics["subgraph_density"] = round(r / max(e, 1), 3)
        metrics["entities"] = [
            {"label": ent.label, "entity_type": ent.entity_type}
            for ent in subgraph.entities
        ]
        metrics["relations"] = [
            {"head": rel.head_text, "relation": rel.relation_type, "tail": rel.tail_text, "score": rel.score}
            for rel in subgraph.relations
        ]
        metrics["chunks"] = [chunk.text for chunk in subgraph.chunks]
    return metrics


def judge_answers(question: str, answers: dict) -> dict:
    """Score each model's answer on relevance and specificity using an LLM as judge.

    Returns a dict mapping each model key to {"relevance": int, "specificity": int} (1–5).
    """
    answer_block = "\n".join(f"[{k}]: {v or 'No answer'}" for k, v in answers.items())
    keys_str = ", ".join(f'"{k}"' for k in answers)
    prompt = (
        "You are an expert medical AI evaluator. Score each answer below on two dimensions:\n"
        "- relevance (1-5): How directly does it answer the specific question?\n"
        "- specificity (1-5): How concrete and clinically detailed is the information?\n\n"
        f"Question: {question}\n\n"
        f"Answers:\n{answer_block}\n\n"
        f"Respond with JSON only, using exactly these keys: {keys_str}.\n"
        'Example: {"baseline": {"relevance": 4, "specificity": 3}, ...}'
    )
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
    }
    try:
        response = requests.post(f"{BASE_URL}/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        return json.loads(response.json()["choices"][0]["message"]["content"])
    except Exception as e:
        print(f"    Judge ERROR: {e}")
        return {}


def query_llm_baseline(questions: list) -> list:
    """Query the LLM directly without any RAG context.

    Returns a list of {"question": ..., "answer": ...} dicts.
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    query_results = []
    for i, question in enumerate(questions, 1):
        print(f"  Q{i}: {question[:70]}...")
        try:
            payload = {
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": question}],
            }
            response = requests.post(f"{BASE_URL}/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            answer = response.json()["choices"][0]["message"]["content"]
            query_results.append({"question": question, "answer": answer})
        except Exception as e:
            print(f"    ERROR: {e}")
            query_results.append({"question": question, "answer": None, "error": str(e)})
    return query_results


def build_graph(config: dict, texts: list) -> dict:
    """Build a RetriCo knowledge graph by ingesting a list of document texts.

    Args:
        config: Dict with model name, pipeline_name, json_output path, and thresholds.
        texts: Document strings to ingest.

    Returns the graph writer stats dict (entity_count, relation_count).
    """
    print(f"\n  Model : {config['model']}")
    print(f"  Output: {config['json_output']}")

    builder = retrico.RetriCoBuilder(name=config["pipeline_name"])
    builder.graph_store(store_type="falkordb_lite", falkordb_lite_graph=config["pipeline_name"])
    builder.chunker(method="sentence")
    builder.relex_gliner(
        model=config["model"],
        entity_labels=ENTITY_LABELS,
        relation_labels=RELATION_LABELS,
        threshold=config["threshold"],
        relation_threshold=config["relation_threshold"],
    )
    builder.graph_writer()
    builder.graph_writer(json_output=config["json_output"])

    executor = builder.build(verbose=True)
    result = executor.run(texts=texts)
    stats = result.get("writer_result") or {}
    print(f"  Entities: {stats.get('entity_count')}, Relations: {stats.get('relation_count')}")
    return stats


def query_model(config: dict, questions: list) -> list:
    """Query a RetriCo knowledge graph and collect answers with retrieval metrics.

    Returns a list of result dicts, one per question, each containing the model's
    answer and retrieval statistics (entities, relations, chunks retrieved).
    """
    search_builder = retrico.RetriCoSearch(name=config["pipeline_name"])
    search_builder.graph_store(store_type="falkordb_lite", falkordb_lite_graph=config["pipeline_name"])
    search_builder.query_parser(method="gliner", labels=ENTITY_LABELS)
    search_builder.retriever(max_hops=2)
    search_builder.chunk_retriever()
    search_builder.reasoner(api_key=API_KEY, model=LLM_MODEL, base_url=BASE_URL)
    executor = search_builder.build()

    query_results = []
    for i, question in enumerate(questions, 1):
        print(f"  Q{i}: {question[:70]}...")
        try:
            result = executor.run(query=question)

            answer = None
            reasoner_output = result.get("reasoner_result")
            if reasoner_output and "result" in reasoner_output:
                answer = reasoner_output["result"].answer

            retrieval_metrics = extract_retrieval_metrics(result)

            query_results.append({
                "question": question,
                "answer": answer,
                "retrieval_metrics": retrieval_metrics,
            })
        except Exception as e:
            print(f"    ERROR: {e}")
            query_results.append({
                "question": question,
                "answer": None,
                "error": str(e),
                "retrieved_info": {},
            })

    return query_results


def print_comparison(questions: list, report: dict) -> None:
    """Print a side-by-side comparison of all model results to stdout.

    Includes graph stats, average retrieval metrics, LLM-as-judge scores,
    and a per-question answer breakdown.
    """
    # --- Graph stats ---
    print("\n" + "=" * 80)
    print("GRAPH STATS COMPARISON")
    print("=" * 80)
    for model_key, data in report.items():
        stats = data.get("graph_stats")
        if stats:
            print(f"  {model_key:<14}: Entities={stats.get('entity_count', 'N/A')}, "
                  f"Relations={stats.get('relation_count', 'N/A')}")
        else:
            print(f"  {model_key:<14}: N/A (no graph)")

    # Average retrieval metrics (RAG models only)
    rag_keys = [k for k, d in report.items() if any("retrieval_metrics" in r for r in d["query_results"])]
    if rag_keys:
        print("\n" + "=" * 80)
        print("AVERAGE RETRIEVAL METRICS PER QUERY (RAG models)")
        print("=" * 80)
        for model_key in rag_keys:
            metrics = [r["retrieval_metrics"] for r in report[model_key]["query_results"] if "retrieval_metrics" in r]
            avg = lambda k: round(sum(m[k] for m in metrics) / len(metrics), 2)
            print(f"  {model_key:<14}: query_entities={avg('query_entities')}, "
                  f"retrieved_entities={avg('retrieved_entities')}, "
                  f"retrieved_relations={avg('retrieved_relations')}, "
                  f"chunks={avg('retrieved_chunks')}, "
                  f"density={avg('subgraph_density')}")

    # Aggregate judge scores
    judge_keys = [k for k, d in report.items() if any("judge_scores" in r for r in d["query_results"])]
    if judge_keys:
        print("\n" + "=" * 80)
        print("AVERAGE LLM-AS-JUDGE SCORES (1–5)")
        print("=" * 80)
        for model_key in judge_keys:
            scores = [r["judge_scores"] for r in report[model_key]["query_results"] if "judge_scores" in r]
            avg_rel = round(sum(s.get("relevance", 0) for s in scores) / len(scores), 2)
            avg_spe = round(sum(s.get("specificity", 0) for s in scores) / len(scores), 2)
            print(f"  {model_key:<14}: relevance={avg_rel}/5, specificity={avg_spe}/5")

    # Per-question breakdown
    # print("\n" + "=" * 80)
    # print("ANSWER COMPARISON")
    # print("=" * 80)
    # for i, question in enumerate(questions):
    #     print(f"\nQ{i + 1}: {question}")
    #     for model_key, data in report.items():
    #         qr = data["query_results"][i]
    #         answer = qr.get("answer") or qr.get("error", "N/A")
    #         scores = qr.get("judge_scores", {})
    #         metrics = qr.get("retrieval_metrics", {})
    #         score_str = (f"  judge: rel={scores.get('relevance','?')} spe={scores.get('specificity','?')}"
    #                      if scores else "")
    #         metrics_str = (f"  retrieved: {metrics.get('retrieved_entities',0)}e "
    #                        f"{metrics.get('retrieved_relations',0)}r "
    #                        f"{metrics.get('retrieved_chunks',0)}chunks "
    #                        f"density={metrics.get('subgraph_density',0)}"
    #                        if metrics else "")
    #         print(f"  [{model_key}]{score_str}{metrics_str}")
    #         print(f"    {answer}\n")


def main():
    # Load documents
    print("Loading documents...")
    texts = extract_texts_for_pipeline(DATA_FILE)
    print(f"Loaded {len(texts)} texts from {DATA_FILE}")

    # Load questions
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        questions = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(questions)} questions from {QUESTIONS_FILE}")

    report = {}

    # Build graphs and query for each model sequentially
    for model_key, config in MODEL_CONFIGS.items():
        print(f"\n{'=' * 60}")
        print(f"PHASE 1 — BUILDING GRAPH: {model_key.upper()}")
        print("=" * 60)
        stats = build_graph(config, texts)

        print(f"\n{'=' * 60}")
        print(f"PHASE 2 — QUERYING: {model_key.upper()}")
        print("=" * 60)
        query_results = query_model(config, questions)

        report[model_key] = {
            "model": config["model"],
            "graph_stats": stats,
            "query_results": query_results,
        }

    # LLM baseline
    print(f"\n{'=' * 60}")
    print("PHASE — LLM BASELINE (no RAG)")
    print("=" * 60)
    llm_results = query_llm_baseline(questions)
    report["llm_baseline"] = {
        "model": LLM_MODEL,
        "graph_stats": None,
        "query_results": llm_results,
    }

    # LLM-as-judge
    print(f"\n{'=' * 60}")
    print("PHASE — LLM-AS-JUDGE EVALUATION")
    print("=" * 60)
    for i, question in enumerate(questions):
        print(f"  Judging Q{i + 1}...")
        answers = {key: data["query_results"][i].get("answer") for key, data in report.items() if key != "llm_baseline"}
        scores = judge_answers(question, answers)
        for key in report:
            if key in scores:
                report[key]["query_results"][i]["judge_scores"] = scores[key]

    # Attach average judge scores to each model entry
    for key, data in report.items():
        scored = [r["judge_scores"] for r in data["query_results"] if "judge_scores" in r]
        if scored:
            data["avg_judge_scores"] = {
                "relevance": round(sum(s.get("relevance", 0) for s in scored) / len(scored), 2),
                "specificity": round(sum(s.get("specificity", 0) for s in scored) / len(scored), 2),
            }

    print_comparison(questions, report)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nFull report saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
