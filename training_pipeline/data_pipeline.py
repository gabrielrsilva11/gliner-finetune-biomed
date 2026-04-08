"""
Loads and merges biomedical NER/RE datasets (ADE, DDI, BioRED, BioREx)
into stratified train/test/eval splits in GLiNER format. Results are
cached as JSON so subsequent runs skip the expensive parsing step.
"""

import os
import json
import glob
import random
from collections import Counter

import spacy
import xml.etree.ElementTree as ET
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from .biorex_preparation import get_biorex_data
from .biored_preparation import get_biored_data

DATA_DIR = "data"
VALID_MODES = ["biorex", "merged_biored"]

CACHE_VERSIONS = {
    "biorex":        "v6",
    "merged_biored": "v7",
}

MIN_TOKENS = 3
DEFAULT_MAX_NEGATIVE_RATIO = 1.5

nlp = spacy.blank("en")


def _build_char_to_token(doc):
    c2t = {}
    for i, token in enumerate(doc):
        for ch in range(token.idx, token.idx + len(token.text)):
            c2t[ch] = i
    return c2t


def get_entity_offsets(full_text, entity_string, index_data):
    if index_data and "start_char" in index_data and len(index_data["start_char"]) > 0:
        return index_data["start_char"][0], index_data["end_char"][0]
    if entity_string:
        start_idx = full_text.find(entity_string)
        if start_idx != -1:
            second = full_text.find(entity_string, start_idx + 1)
            if second != -1:
                return None, None
            return start_idx, start_idx + len(entity_string)
    return None, None


def get_ade_data():
    """Load and parse the ADE Corpus v2 drug–adverse-effect relation dataset.

    Returns a list of GLiNER-format samples with Drug and Adverse Effect spans
    and 'causes' relations.
    """
    print("Loading ADE Corpus from Hugging Face...")
    dataset = load_dataset("ade_corpus_v2", "Ade_corpus_v2_drug_ade_relation")

    grouped_sentences = {}
    for row in dataset["train"]:
        grouped_sentences.setdefault(row["text"], []).append(row)

    formatted_data = []
    skipped_ambiguous = 0
    print("Parsing ADE data...")

    for text, relations in grouped_sentences.items():
        doc    = nlp(text)
        tokens = [token.text for token in doc]
        c2t    = _build_char_to_token(doc)

        ner_list       = []
        relations_list = []
        span_to_idx    = {}

        def add_entity(start_char, end_char, label):
            start_tok = c2t.get(start_char) or c2t.get(start_char + 1)
            end_tok   = c2t.get(end_char - 1) or c2t.get(end_char - 2)
            if start_tok is None or end_tok is None:
                return None
            key = (start_tok, end_tok, label)
            if key not in span_to_idx:
                span_to_idx[key] = len(ner_list)
                ner_list.append([start_tok, end_tok, label])
            return span_to_idx[key]

        for rel in relations:
            drug_start, drug_end = get_entity_offsets(
                text, rel.get("drug", ""), rel.get("indexes", {}).get("drug", {})
            )
            effect_start, effect_end = get_entity_offsets(
                text, rel.get("effect", ""), rel.get("indexes", {}).get("effect", {})
            )
            if drug_start is None or effect_start is None:
                skipped_ambiguous += 1
                continue
            drug_idx   = add_entity(drug_start,   drug_end,   "Drug")
            effect_idx = add_entity(effect_start, effect_end, "Adverse Effect")
            if drug_idx is not None and effect_idx is not None:
                rel_tuple = [drug_idx, effect_idx, "causes"]
                if rel_tuple not in relations_list:
                    relations_list.append(rel_tuple)

        if ner_list:
            formatted_data.append({
                "tokenized_text": tokens,
                "ner":            ner_list,
                "relations":      relations_list,
            })

    if skipped_ambiguous:
        print(f"  ADE: {skipped_ambiguous} relations skipped (ambiguous string-find fallback)")

    return formatted_data


DDI_RELATION_MAP = {
    "effect":    "has effect on",
    "mechanism": "alters mechanism of",
    "advise":    "clinical advice",
    "int":       "interacts with",
}

DDI_ENTITY_MAP = {
    "drug":   "Drug",
    "brand":  "Brand",
    "group":  "Group",
    "drug_n": "Non-Human Drug",
}


def get_ddi_data(ddi_dir_path):
    """Parse the DDI Corpus XML files into GLiNER-format samples.

    Args:
        ddi_dir_path: Path to the DDI Corpus directory containing XML files.

    Returns a list of GLiNER-format samples with drug entity spans and
    drug–drug interaction relations.
    """
    print(f"Parsing DDI XML files from {ddi_dir_path}...")
    xml_files      = glob.glob(os.path.join(ddi_dir_path, "**/*.xml"), recursive=True)
    formatted_data = []
    multi_span_warnings = 0

    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        for sentence in root.findall(".//sentence"):
            text = sentence.get("text")
            if not text:
                continue

            doc    = nlp(text)
            tokens = [t.text for t in doc]
            c2t    = _build_char_to_token(doc)

            ner_list         = []
            id_to_gliner_idx = {}

            for entity in sentence.findall(".//entity"):
                eid      = entity.get("id")
                offsets  = entity.get("charOffset")
                raw_type = entity.get("type")
                if not offsets or not raw_type:
                    continue

                mapped_label = DDI_ENTITY_MAP.get(raw_type, "Drug")

                spans = offsets.split(";")
                if len(spans) > 1:
                    multi_span_warnings += 1

                first_span = spans[0]
                start_str, end_str = first_span.split("-")
                start_char, end_char = int(start_str), int(end_str)

                start_tok = c2t.get(start_char) or c2t.get(start_char + 1) or c2t.get(start_char - 1)
                end_tok   = c2t.get(end_char)   or c2t.get(end_char - 1)   or c2t.get(end_char + 1)

                if start_tok is not None and end_tok is not None:
                    id_to_gliner_idx[eid] = len(ner_list)
                    ner_list.append([start_tok, end_tok, mapped_label])

            relations_list = []
            for pair in sentence.findall(".//pair"):
                if pair.get("ddi") != "true":
                    continue
                raw_type = pair.get("type")
                if raw_type is None:
                    continue
                e1, e2     = pair.get("e1"), pair.get("e2")
                mapped_rel = DDI_RELATION_MAP.get(raw_type, raw_type)
                if e1 in id_to_gliner_idx and e2 in id_to_gliner_idx:
                    relations_list.append([
                        id_to_gliner_idx[e1],
                        id_to_gliner_idx[e2],
                        mapped_rel,
                    ])

            if ner_list:
                formatted_data.append({
                    "tokenized_text": tokens,
                    "ner":            ner_list,
                    "relations":      relations_list,
                })

    if multi_span_warnings:
        print(f"  DDI: {multi_span_warnings} entities had discontinuous spans (first span used)")

    return formatted_data


def validate_samples(data: list, source: str = "") -> list:
    """Remove malformed samples from a dataset.

    Drops samples that are too short, have out-of-bounds NER spans, or have
    relation indices that point to non-existent entities.
    """
    valid   = []
    dropped = {"too_short": 0, "ner_oob": 0, "rel_oob": 0}

    for sample in data:
        tokens   = sample.get("tokenized_text", [])
        ner_list = sample.get("ner", [])
        rel_list = sample.get("relations", [])
        n_tok    = len(tokens)
        n_ner    = len(ner_list)

        if n_tok < MIN_TOKENS:
            dropped["too_short"] += 1
            continue
        if not all(0 <= s <= e < n_tok for s, e, *_ in ner_list):
            dropped["ner_oob"] += 1
            continue
        if not all(0 <= h < n_ner and 0 <= t < n_ner for h, t, *_ in rel_list):
            dropped["rel_oob"] += 1
            continue

        valid.append(sample)

    total = sum(dropped.values())
    tag   = f" [{source}]" if source else ""
    if total:
        print(
            f"  validate_samples{tag}: removed {total} malformed samples "
            f"(too_short={dropped['too_short']}, ner_oob={dropped['ner_oob']}, "
            f"rel_oob={dropped['rel_oob']})"
        )
    return valid


def cap_negatives(data: list, max_ratio: float, source: str = "") -> list:
    """Subsample negative examples to keep the positive/negative ratio bounded.

    Positive samples have at least one relation; negatives have none. Returns
    the data unchanged if the ratio is already within the limit.
    """
    positives = [s for s in data if s["relations"]]
    negatives = [s for s in data if not s["relations"]]
    max_neg   = int(len(positives) * max_ratio)
    tag       = f" [{source}]" if source else ""

    if len(negatives) <= max_neg:
        print(
            f"  cap_negatives{tag}: {len(positives)} pos, "
            f"{len(negatives)} neg — ratio OK, no subsampling."
        )
        return data

    random.seed(42)
    negatives = random.sample(negatives, max_neg)
    result    = positives + negatives
    random.shuffle(result)
    print(
        f"  cap_negatives{tag}: {len(positives)} pos, "
        f"neg {len(data) - len(positives)} -> {max_neg} (ratio {max_ratio:.1f}x)."
    )
    return result


def _print_label_frequencies(data: list, tag: str = ""):
    ent_counts = Counter()
    rel_counts = Counter()
    for sample in data:
        for _, _, label in sample["ner"]:
            ent_counts[label] += 1
        for _, _, rtype in sample["relations"]:
            rel_counts[rtype] += 1

    header = f"Label frequencies{f' [{tag}]' if tag else ''}"
    print(f"\n  {header}")
    print(f"  {'Entity':<25} {'Count':>6}")
    print(f"  {'-'*32}")
    for label, cnt in ent_counts.most_common():
        print(f"  {str(label or '<None>'):<25} {cnt:>6}")
    print(f"  {'Relation':<25} {'Count':>6}")
    print(f"  {'-'*32}")
    for label, cnt in rel_counts.most_common():
        print(f"  {str(label or '<None>'):<25} {cnt:>6}")
    print()


def prepare_merged_data(
    mode="merged_biored",
    ddi_dir_path="data/DDICorpus",
    biorex_dir_path="data/biorex",
    max_negative_ratio=DEFAULT_MAX_NEGATIVE_RATIO,
):
    """Load or build train/test/eval splits for the specified dataset mode.

    On first call, parses the raw corpora, validates and balances samples,
    then splits stratified by source and caches to JSON. Subsequent calls
    load directly from the cache.

    Args:
        mode: "merged_biored" (ADE + DDI + BioRED) or "biorex".
        ddi_dir_path: Path to DDI Corpus XML directory.
        biorex_dir_path: Path to BioREx PubTator directory.
        max_negative_ratio: Maximum negatives-to-positives ratio after capping.

    Returns a dict with "train", "test", and "eval" keys.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown mode: {mode!r}. Choose from: {VALID_MODES}")

    cache_ver = CACHE_VERSIONS[mode]

    # Use mode-specific cache files
    split_files = {
        "train": os.path.join(DATA_DIR, f"{mode}_{cache_ver}_train.json"),
        "test":  os.path.join(DATA_DIR, f"{mode}_{cache_ver}_test.json"),
        "eval":  os.path.join(DATA_DIR, f"{mode}_{cache_ver}_eval.json"),
    }

    if all(os.path.exists(f) for f in split_files.values()):
        print(f"Loading cached {mode} splits ({cache_ver}) from {DATA_DIR}/...")
        splits = {}
        for name, filepath in split_files.items():
            with open(filepath) as f:
                splits[name] = json.load(f)
        return splits

    sources = {}

    if mode == "merged_biored":
        print("\n--- ADE ---")
        sources["ade"] = validate_samples(get_ade_data(), source="ADE")

        print("\n--- DDI ---")
        sources["ddi"] = validate_samples(get_ddi_data(ddi_dir_path), source="DDI")

        print("\n--- BioRED ---")
        sources["biored"] = validate_samples(get_biored_data(), source="BioRED")

    elif mode == "biorex":
        print("\n--- BioREx ---")
        sources["biorex"] = validate_samples(get_biorex_data(biorex_dir_path), source="BioREx")

    if max_negative_ratio is not None:
        for key in sources:
            sources[key] = cap_negatives(
                sources[key], max_negative_ratio, source=key.upper(),
            )

    for key, data in sources.items():
        _print_label_frequencies(data, key.upper())

    for key, data in sources.items():
        for s in data:
            s["_source"] = key

    merged_data = []
    for data in sources.values():
        merged_data.extend(data)

    n_pos = sum(1 for s in merged_data if s["relations"])
    n_neg = len(merged_data) - n_pos
    source_summary = " | ".join(
        f"{k.upper()}: {len(v)}" for k, v in sources.items()
    )
    print(
        f"\nDataset ({mode}): {len(merged_data)} samples total  ({source_summary})\n"
        f"  Positive (has relations) : {n_pos}\n"
        f"  Negative (no relations)  : {n_neg}\n"
        + (f"  Ratio neg/pos            : {n_neg / n_pos:.2f}" if n_pos else "")
    )

    _print_label_frequencies(merged_data, mode)

    source_labels = [s["_source"] for s in merged_data]
    train_data, temp_data = train_test_split(
        merged_data, test_size=0.30, random_state=42, stratify=source_labels,
    )
    temp_sources = [s["_source"] for s in temp_data]
    test_data, eval_data = train_test_split(
        temp_data, test_size=0.50, random_state=42, stratify=temp_sources,
    )

    for split in (train_data, test_data, eval_data):
        for s in split:
            s.pop("_source", None)

    splits = {"train": train_data, "test": test_data, "eval": eval_data}

    os.makedirs(DATA_DIR, exist_ok=True)
    for name, data in splits.items():
        with open(split_files[name], "w") as f:
            json.dump(data, f)

    print("Successfully created and saved splits.")
    return splits