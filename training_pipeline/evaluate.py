"""
Evaluates a baseline and a fine-tuned GLiNER model on the test set,
reporting NER and RE precision/recall/F1 with a per-relation-type breakdown.
"""

import sys
import os

os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.getcwd())

import torch
import pandas as pd
from collections import defaultdict
from gliner import GLiNER
from tqdm import tqdm

from training_pipeline.data_pipeline import prepare_merged_data
from training_pipeline.re_callback import DATASET_CONFIG

# biorex or merged_biored
DATASET = "biorex"
ENTITY_LABELS    = DATASET_CONFIG[DATASET]["entity_labels"]
RELATION_LABELS  = DATASET_CONFIG[DATASET]["relation_labels"]
THRESHOLD          = 0.4
RELATION_THRESHOLD = 0.5
FINETUNED_MODEL = "grsilva/gliner-relex-biomedical-lora"
BASELINE_MODEL = "knowledgator/gliner-relex-large-v1.0"

def _prf(tp, fp, fn):
    p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def _build_char_to_token(tokens):
    c2t = {}
    pos = 0
    for i, tok in enumerate(tokens):
        for j in range(len(tok)):
            c2t[pos + j] = i
        pos += len(tok) + 1
    return c2t


def _pred_to_token_span(c2t, entity_dict):
    start_char = entity_dict["start"]
    end_char   = entity_dict["end"] - 1  # end is exclusive

    start_tok = c2t.get(start_char)
    end_tok   = c2t.get(end_char)

    if start_tok is None:
        start_tok = c2t.get(start_char + 1) or c2t.get(start_char - 1)
    if end_tok is None:
        end_tok = c2t.get(end_char - 1) or c2t.get(end_char + 1)

    return start_tok, end_tok


def evaluate_model(model, test_data, entity_labels, relation_labels):
    """Run NER and RE evaluation for a single model over the full test set.

    Args:
        model: Loaded GLiNER model.
        test_data: List of GLiNER-format samples with tokenized_text, ner, and relations keys.
        entity_labels: Entity label strings passed to model.inference.
        relation_labels: Relation label strings passed to model.inference.

    Returns:
        df_summary: DataFrame with aggregate NER and RE precision/recall/F1.
        df_per_type: DataFrame with per-relation-type metrics, or None if no relations seen.
    """
    ner_tp = ner_fp = ner_fn = 0
    re_tp  = re_fp  = re_fn  = 0

    per_type_tp = defaultdict(int)
    per_type_fp = defaultdict(int)
    per_type_fn = defaultdict(int)

    pbar = tqdm(test_data, desc="Evaluating", unit="sample")

    for i, sample in enumerate(pbar):
        tokens = sample["tokenized_text"]
        text   = " ".join(tokens)
        c2t    = _build_char_to_token(tokens)

        # Ground-truth sets (token-index based)
        gt_ner = {
            (s, e, lbl)
            for s, e, lbl in sample["ner"]
        }
        gt_rels = set()
        for h, t, rel in sample["relations"]:
            head_ner = sample["ner"][h]
            tail_ner = sample["ner"][t]
            gt_rels.add((
                head_ner[0], head_ner[1],
                tail_ner[0], tail_ner[1],
                rel,
            ))

        # Inference
        try:
            entities, relations = model.inference(
                texts=[text],
                labels=entity_labels,
                relations=relation_labels,
                threshold=THRESHOLD,
                relation_threshold=RELATION_THRESHOLD,
                return_relations=True,
                flat_ner=True,
            )
        except Exception as e:
            print(f"Warning: inference failed for sample {i}: {e}")
            ner_fn += len(gt_ner)
            re_fn  += len(gt_rels)
            continue

        # Predicted entities - token spans
        pred_ner = set()
        for ent in entities[0]:
            s_tok, e_tok = _pred_to_token_span(c2t, ent)
            if s_tok is not None and e_tok is not None:
                pred_ner.add((s_tok, e_tok, ent["label"]))

        # Predicted relations - token spans
        pred_rels = set()
        for rel in relations[0]:
            h_s, h_e = _pred_to_token_span(c2t, rel["head"])
            t_s, t_e = _pred_to_token_span(c2t, rel["tail"])
            if all(x is not None for x in (h_s, h_e, t_s, t_e)):
                pred_rels.add((h_s, h_e, t_s, t_e, rel["relation"]))

        # NER count
        ner_tp += len(gt_ner  & pred_ner)
        ner_fp += len(pred_ner - gt_ner)
        ner_fn += len(gt_ner  - pred_ner)

        # RE count
        re_tp += len(gt_rels  & pred_rels)
        re_fp += len(pred_rels - gt_rels)
        re_fn += len(gt_rels  - pred_rels)

        # RE count per type
        all_types = {r[-1] for r in gt_rels} | {r[-1] for r in pred_rels}
        for rtype in all_types:
            gt_t   = {r for r in gt_rels   if r[-1] == rtype}
            pred_t = {r for r in pred_rels if r[-1] == rtype}
            per_type_tp[rtype] += len(gt_t & pred_t)
            per_type_fp[rtype] += len(pred_t - gt_t)
            per_type_fn[rtype] += len(gt_t - pred_t)

        if i % 10 == 0:
            _, _, live_f1 = _prf(re_tp, re_fp, re_fn)
            pbar.set_postfix({"Live_RE_F1": f"{live_f1:.3f}"})

    # Aggregate metrics
    ner_p, ner_r, ner_f1 = _prf(ner_tp, ner_fp, ner_fn)
    re_p,  re_r,  re_f1  = _prf(re_tp,  re_fp,  re_fn)

    df_summary = pd.DataFrame(
        {
            "Precision": [ner_p,  re_p],
            "Recall":    [ner_r,  re_r],
            "F1":        [ner_f1, re_f1],
        },
        index=["NER", "RE"],
    )

    # Per-relation-type breakdown
    all_seen = sorted(set(per_type_tp) | set(per_type_fp) | set(per_type_fn))
    rows = []
    for rtype in all_seen:
        p, r, f = _prf(per_type_tp[rtype], per_type_fp[rtype], per_type_fn[rtype])
        rows.append({
            "Relation": rtype,
            "Precision": p,
            "Recall": r,
            "F1": f,
            "Support": per_type_tp[rtype] + per_type_fn[rtype],
        })
    df_per_type = pd.DataFrame(rows).set_index("Relation") if rows else None

    return df_summary, df_per_type


def main():
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Device: {device}")
    print(f"Dataset config : {DATASET}")
    print(f"Entity labels  : {ENTITY_LABELS}")
    print(f"Relation labels: {RELATION_LABELS}\n")

    test_data = prepare_merged_data(mode=DATASET)["test"]
    print(f"Loaded {len(test_data)} test samples.\n")

    sep = "=" * 55

    print(f"Evaluating baseline model ({BASELINE_MODEL})...")
    baseline_model = GLiNER.from_pretrained(BASELINE_MODEL).to(device)
    baseline_model.eval()
    df_base, df_base_per = evaluate_model(baseline_model, test_data, ENTITY_LABELS, RELATION_LABELS)
    del baseline_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


    print(f"\nEvaluating fine-tuned model ({FINETUNED_MODEL})...")
    finetuned_model = GLiNER.from_pretrained(FINETUNED_MODEL).to(device)
    
    finetuned_model.eval()
    df_ft, df_ft_per = evaluate_model(finetuned_model, test_data, ENTITY_LABELS, RELATION_LABELS)
    del finetuned_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n{sep}")
    print("BASELINE MODEL METRICS")
    print(sep)
    print(df_base.round(4))
    if df_base_per is not None:
        print("\nPer-relation breakdown:")
        print(df_base_per.round(4))

    print(f"\n{sep}")
    print(f"FINE-TUNED MODEL METRICS")
    print(sep)
    print(df_ft.round(4))
    if df_ft_per is not None:
        print("\nPer-relation breakdown:")
        print(df_ft_per.round(4))

    print(f"\n{sep}")
    print("DELTA  (Fine-Tuned − Baseline)")
    print(sep)
    df_delta = df_ft - df_base
    print(df_delta.map(lambda x: f"{x:+.4f}"))


if __name__ == "__main__":
    main()