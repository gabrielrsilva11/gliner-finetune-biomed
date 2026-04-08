"""
Sweeps NER and RE confidence thresholds on the test set to find the
combination that maximises RE F1. Predictions at each NER threshold are
cached in memory so only the outer loop triggers model inference.
"""

import sys
import os

os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.getcwd())

import torch
from gliner import GLiNER
from tqdm import tqdm

from training_pipeline.data_pipeline import prepare_merged_data
from training_pipeline.re_callback import DATASET_CONFIG

DATASET = "merged_biored"
ENTITY_LABELS   = DATASET_CONFIG[DATASET]["entity_labels"]
RELATION_LABELS = DATASET_CONFIG[DATASET]["relation_labels"]
MODEL_PATH = "grsilva/gliner-relex-biomedical-lora"
# MODEL_PATH = "knowledgator/gliner-relex-large-v1.0"

NER_THRESHOLDS = [0.2, 0.3, 0.35, 0.4, 0.45, 0.5]
RE_THRESHOLDS  = [0.3, 0.4, 0.5]


def _prf(tp, fp, fn):
    p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def _build_c2t(tokens):
    c2t = {}
    pos = 0
    for i, tok in enumerate(tokens):
        for j in range(len(tok)):
            c2t[pos + j] = i
        pos += len(tok) + 1
    return c2t


def _to_tok(c2t, ent):
    sc = ent["start"]
    ec = ent["end"] - 1
    st = c2t.get(sc) or c2t.get(sc + 1) or c2t.get(sc - 1)
    et = c2t.get(ec) or c2t.get(ec - 1) or c2t.get(ec + 1)
    return st, et


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = GLiNER.from_pretrained(MODEL_PATH).to(device)
    model.eval()

    test_data = prepare_merged_data()["test"]
    print(f"Model: {MODEL_PATH}")
    print(f"Test samples: {len(test_data)}\n")

    # Precompute ground truth
    gt_cache = []
    for sample in test_data:
        tokens = sample["tokenized_text"]
        text = " ".join(tokens)
        c2t = _build_c2t(tokens)

        gt_ner = {(s, e, l) for s, e, l in sample["ner"]}
        gt_rels = set()
        for h, t, rel in sample["relations"]:
            hn = sample["ner"][h]
            tn = sample["ner"][t]
            gt_rels.add((hn[0], hn[1], tn[0], tn[1], rel))

        gt_cache.append((text, c2t, gt_ner, gt_rels))

    print(f"{'NER_t':>6} {'RE_t':>6} | {'NER_P':>7} {'NER_R':>7} {'NER_F1':>7} | {'RE_P':>7} {'RE_R':>7} {'RE_F1':>7}")
    print("-" * 75)

    best_re_f1 = 0
    best_combo = None

    for ner_t in NER_THRESHOLDS:
        # Cache predictions at this NER threshold
        all_preds = []
        for text, c2t, _, _ in tqdm(gt_cache, desc=f"NER_t={ner_t}", leave=False):
            try:
                ents, rels = model.inference(
                    texts=[text],
                    labels=ENTITY_LABELS,
                    relations=RELATION_LABELS,
                    threshold=ner_t,
                    relation_threshold=0.01,  # get all, filter later
                    return_relations=True,
                    flat_ner=True,
                )
                all_preds.append((ents[0], rels[0]))
            except Exception:
                all_preds.append(([], []))

        for re_t in RE_THRESHOLDS:
            ner_tp = ner_fp = ner_fn = 0
            re_tp = re_fp = re_fn = 0

            for i, (text, c2t, gt_ner, gt_rels) in enumerate(gt_cache):
                pred_ents, pred_rels_raw = all_preds[i]

                # NER
                pred_ner = set()
                for ent in pred_ents:
                    st, et = _to_tok(c2t, ent)
                    if st is not None and et is not None:
                        pred_ner.add((st, et, ent["label"]))

                ner_tp += len(gt_ner & pred_ner)
                ner_fp += len(pred_ner - gt_ner)
                ner_fn += len(gt_ner - pred_ner)

                # RE (filter by RE threshold)
                pred_rels = set()
                for rel in pred_rels_raw:
                    if rel.get("score", 1.0) < re_t:
                        continue
                    hs, he = _to_tok(c2t, rel["head"])
                    ts, te = _to_tok(c2t, rel["tail"])
                    if all(x is not None for x in (hs, he, ts, te)):
                        pred_rels.add((hs, he, ts, te, rel["relation"]))

                re_tp += len(gt_rels & pred_rels)
                re_fp += len(pred_rels - gt_rels)
                re_fn += len(gt_rels - pred_rels)

            np, nr, nf = _prf(ner_tp, ner_fp, ner_fn)
            rp, rr, rf = _prf(re_tp, re_fp, re_fn)

            marker = ""
            if rf > best_re_f1:
                best_re_f1 = rf
                best_combo = (ner_t, re_t)
                marker = " <-- best"

            print(f"{ner_t:>6.2f} {re_t:>6.2f} | {np:>7.4f} {nr:>7.4f} {nf:>7.4f} | {rp:>7.4f} {rr:>7.4f} {rf:>7.4f}{marker}")

    print(f"\nBest RE F1: {best_re_f1:.4f} at NER_t={best_combo[0]}, RE_t={best_combo[1]}")


if __name__ == "__main__":
    main()