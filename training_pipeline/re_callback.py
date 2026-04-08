from collections import defaultdict
from tqdm import tqdm
from transformers import TrainerCallback
from .biored_preparation import RELATION_LABELS as BIORED_RELATION_LABELS

DATASET_CONFIG = {
    "merged_biored": {
        "entity_labels": [
            "Drug", "Brand", "Group", "Non-Human Drug", "Adverse Effect",
            "Gene", "Disease", "Chemical", "Variant", "Species", "Cell Line",
        ],
        "relation_labels": [
            "causes",
            "effect",
            "mechanism",
            "advice",
            "interacts",
            *BIORED_RELATION_LABELS,
        ],
    },
    "biorex": {
        "entity_labels": [
            "Gene", "Disease", "Chemical", "Variant",
            "Species", "Cell Line",
        ],
        "relation_labels": [
            "Association", "causes", "Positive Correlation",
            "Negative Correlation", "Bind", "Cotreatment",
            "Comparison", "Conversion", "Drug Interaction",
            "inhibitor", "activator", "agonist", "antagonist",
            "substrate", "product of", "part of",
            "direct regulator", "indirect downregulator",
            "indirect upregulator",
        ],
    },
}

VALID_DATASETS = list(DATASET_CONFIG.keys())

def _prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


class REOptimizationCallback(TrainerCallback):
    def __init__(
        self,
        eval_data,
        dataset: str = "merged_biored",
        threshold: float = 0.2,
        relation_threshold: float = 0.5,
        composite_weight: float = 0.6,
        batch_size: int = 16,
    ):
        self.threshold          = threshold
        self.relation_threshold = relation_threshold
        self.composite_weight   = composite_weight
        self.batch_size         = batch_size

        if dataset not in DATASET_CONFIG:
            raise ValueError(
                f"Unknown dataset: {dataset!r}. Choose one of: {VALID_DATASETS}"
            )
        cfg = DATASET_CONFIG[dataset]
        self.entity_labels   = cfg["entity_labels"]
        self.relation_labels = cfg["relation_labels"]

        # pre-compute everything derived from eval_data once at init.
        # texts, c2t maps, gt_entities and gt_rels are all fixed across epochs
        # so there is no reason to recompute them inside on_evaluate.
        self.texts       = []
        self.c2t_maps    = []
        self.gt_entities = []
        self.gt_rels     = []

        for sample in eval_data:
            tokens = sample["tokenized_text"]
            text   = " ".join(tokens)
            c2t    = self._build_char_to_token(tokens)

            gt_ents = {
                (s, e, label)
                for s, e, label in sample["ner"]
            }

            gt_rel_set = set()
            for h, t, rel in sample["relations"]:
                head_ner = sample["ner"][h]
                tail_ner = sample["ner"][t]
                gt_rel_set.add((
                    head_ner[0], head_ner[1],
                    tail_ner[0], tail_ner[1],
                    rel,
                ))

            self.texts.append(text)
            self.c2t_maps.append(c2t)
            self.gt_entities.append(gt_ents)
            self.gt_rels.append(gt_rel_set)

    def _build_char_to_token(self, tokens):
        c2t = {}
        pos = 0
        for i, tok in enumerate(tokens):
            for j in range(len(tok)):
                c2t[pos + j] = i
            pos += len(tok) + 1   # +1 for the joining space
        return c2t

    def _pred_entity_to_token_span(self, c2t, entity_dict):
        start_char = entity_dict["start"]
        end_char   = entity_dict["end"] - 1   # end is exclusive in predictions

        start_tok = c2t.get(start_char)
        end_tok   = c2t.get(end_char)

        # Widened fallback
        if start_tok is None:
            for delta in (1, -1, 2, -2):
                start_tok = c2t.get(start_char + delta)
                if start_tok is not None:
                    break
        if end_tok is None:
            for delta in (-1, 1, -2, 2):
                end_tok = c2t.get(end_char + delta)
                if end_tok is not None:
                    break

        return start_tok, end_tok

    def on_evaluate(self, args, state, control, model, metrics, **kwargs):
        import torch
        print("\n[Callback] Calculating NER + RE metrics…")
        was_training = model.training
        model.eval()

        # NER Counters
        ner_tp = ner_fp = ner_fn = 0

        # RE counters
        re_tp = re_fp = re_fn = 0
        per_type_tp = defaultdict(int)
        per_type_fp = defaultdict(int)
        per_type_fn = defaultdict(int)

        n = len(self.texts)

        # Batch Inference for faster eval
        n_batches = (n + self.batch_size - 1) // self.batch_size

        with torch.no_grad():
            pbar = tqdm(
                range(0, n, self.batch_size),
                total=n_batches,
                desc="[Callback] Eval",
                unit="batch",
                leave=False,
            )
            for batch_start in pbar:
                batch_end = min(batch_start + self.batch_size, n)
                pbar.set_postfix(samples=f"{batch_end}/{n}")

                batch_texts       = self.texts[batch_start:batch_end]
                batch_c2t         = self.c2t_maps[batch_start:batch_end]
                batch_gt_entities = self.gt_entities[batch_start:batch_end]
                batch_gt_rels     = self.gt_rels[batch_start:batch_end]

                try:
                    entities_batch, relations_batch = model.inference(
                        texts=batch_texts,
                        labels=self.entity_labels,
                        relations=self.relation_labels,
                        threshold=self.threshold,
                        relation_threshold=self.relation_threshold,
                        return_relations=True,
                        flat_ner=True,
                    )
                except Exception:
                    entities_batch  = [[] for _ in batch_texts]
                    relations_batch = [[] for _ in batch_texts]

                for i in range(len(batch_texts)):
                    c2t         = batch_c2t[i]
                    gt_entities = batch_gt_entities[i]
                    gt_rels     = batch_gt_rels[i]

                    # Map predicted entities to token spans
                    pred_entities = set()
                    for ent in entities_batch[i]:
                        s_tok, e_tok = self._pred_entity_to_token_span(c2t, ent)
                        if s_tok is not None and e_tok is not None:
                            pred_entities.add((s_tok, e_tok, ent["label"]))

                    # NER evaluation
                    ner_tp += len(gt_entities & pred_entities)
                    ner_fp += len(pred_entities - gt_entities)
                    ner_fn += len(gt_entities - pred_entities)

                    # Token-index-based RE matching
                    pred_rels = set()
                    for rel in relations_batch[i]:
                        h_start, h_end = self._pred_entity_to_token_span(c2t, rel["head"])
                        t_start, t_end = self._pred_entity_to_token_span(c2t, rel["tail"])
                        if all(x is not None for x in (h_start, h_end, t_start, t_end)):
                            pred_rels.add((h_start, h_end, t_start, t_end, rel["relation"]))

                    # Aggregate RE
                    re_tp += len(gt_rels & pred_rels)
                    re_fp += len(pred_rels - gt_rels)
                    re_fn += len(gt_rels - pred_rels)

                    # Per-type RE
                    all_types = {r[-1] for r in gt_rels} | {r[-1] for r in pred_rels}
                    for rtype in all_types:
                        gt_t   = {r for r in gt_rels   if r[-1] == rtype}
                        pred_t = {r for r in pred_rels if r[-1] == rtype}
                        per_type_tp[rtype] += len(gt_t & pred_t)
                        per_type_fp[rtype] += len(pred_t - gt_t)
                        per_type_fn[rtype] += len(gt_t - pred_t)

        ner_p, ner_r, ner_f1 = _prf(ner_tp, ner_fp, ner_fn)
        metrics["eval_ner_f1"] = ner_f1
        print(
            f"[Callback] NER F1: {ner_f1:.4f}  "
            f"(P: {ner_p:.4f}, R: {ner_r:.4f})"
        )

        re_p, re_r, re_f1 = _prf(re_tp, re_fp, re_fn)
        metrics["eval_re_f1"] = re_f1
        print(
            f"[Callback] RE  F1: {re_f1:.4f}  "
            f"(P: {re_p:.4f}, R: {re_r:.4f})"
        )

        w = self.composite_weight
        composite = (1.0 - w) * ner_f1 + w * re_f1
        metrics["eval_composite"] = composite
        print(
            f"[Callback] Composite ({w:.0%} RE / {1-w:.0%} NER): {composite:.4f}"
        )

        all_seen_types = sorted(
            set(per_type_tp) | set(per_type_fp) | set(per_type_fn)
        )

        known_types = sorted(set(self.relation_labels) | set(all_seen_types))

        if known_types:
            print(f"\n  {'Relation Type':<30} {'P':>7} {'R':>7} {'F1':>7} {'Support':>8}")
            print(f"  {'-'*59}")
            for rtype in known_types:
                tp_t = per_type_tp[rtype]
                fp_t = per_type_fp[rtype]
                fn_t = per_type_fn[rtype]
                p_t, r_t, f_t = _prf(tp_t, fp_t, fn_t)
                support = tp_t + fn_t
                print(f"  {rtype:<30} {p_t:>7.4f} {r_t:>7.4f} {f_t:>7.4f} {support:>8}")
            print()

        if was_training:
            model.train()

# Basic collator to wrap around the Gliner span collator and drop any
# examples that are wrong in a batch.
class SafeCollator:
    def __init__(self, inner):
        self.inner = inner
        self.dropped = 0

    def __call__(self, batch):
        try:
            return self.inner(batch)
        except RuntimeError as e:
            if "negative output size" not in str(e) and "zero" not in str(e).lower():
                raise
            good_samples = []
            for sample in batch:
                try:
                    self.inner([sample])
                    good_samples.append(sample)
                except RuntimeError:
                    self.dropped += 1
                    continue
            if not good_samples:
                raise RuntimeError("All samples in this batch failed collation. Check dataset.")
            return self.inner(good_samples)