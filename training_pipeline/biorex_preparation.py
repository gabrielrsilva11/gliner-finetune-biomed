"""
Parses BioREx sub-datasets from PubTator files into GLiNER format.
Covers multiple source corpora (CDR, DrugProt, DisGeNET, PPI datasets,
PharmGKB) with configurable coreference resolution modes.
"""

import os
import glob
import random
import spacy
from collections import Counter, defaultdict
from typing import Literal

BIOREX_DIR = os.path.join("data", "biorex")

ENTITY_TYPE_MAP = {
    # BioRED / ncbi_relation
    "GeneOrGeneProduct":          "Gene",
    "DiseaseOrPhenotypicFeature": "Disease",
    "ChemicalEntity":             "Chemical",
    "SequenceVariant":            "Variant",
    "OrganismTaxon":              "Species",
    "CellLine":                   "Cell Line",
    # CDR
    "Chemical":                   "Chemical",
    "Disease":                    "Disease",
    # DrugProt
    "CHEMICAL":                   "Chemical",
    "GENE":                       "Gene",
    # DDI
    "drug":                       "Drug",
    "brand":                      "Brand",
    "group":                      "Group",
    "drug_n":                     "Non-Human Drug",
    # PPI datasets
    "Gene":                       "Gene",
    "Protein":                    "Gene",  # often used in PPI datasets
    # PharmGKB / DisGeNET variant subtypes
    "DNAMutation":                "Variant",
    "ProteinMutation":            "Variant",
    "SNP":                        "Variant",
    "DNAAcidChange":              "Variant",   # PharmGKB
    "ProteinAcidChange":          "Variant",   # PharmGKB / DisGeNET
    "DNAAllele":                  "Variant",   # PharmGKB
    "ProteinAllele":              "Variant",   # PharmGKB
}

RELATION_TYPE_MAP = {
    # BioRED
    "Association":           "Association",
    "Positive_Correlation":  "Positive Correlation",
    "Negative_Correlation":  "Negative Correlation",
    "Bind":                  "Bind",
    "Cotreatment":           "Cotreatment",
    "Comparison":            "Comparison",
    "Conversion":            "Conversion",
    "Drug_Interaction":      "Drug Interaction",
    # CDR
    "CID":                   "causes",
    # DrugProt
    "INHIBITOR":             "inhibitor",
    "ACTIVATOR":             "activator",
    "AGONIST":               "agonist",
    "ANTAGONIST":            "antagonist",
    "SUBSTRATE":             "substrate",
    "PRODUCT-OF":            "product of",
    "PART-OF":               "part of",
    "INDIRECT-DOWNREGULATOR":"indirect downregulator",
    "INDIRECT-UPREGULATOR":  "indirect upregulator",
    "DIRECT-REGULATOR":      "direct regulator",
    # PPI Datasets
    "PPI":                   "interacts with",
    "UNDEFINED":             "interacts with",
}

PUBTATOR_SOURCES = [
    ("ncbi_relation",[
        "ncbi_relation/Train.PubTator",
        "ncbi_relation/Dev.PubTator",
        "ncbi_relation/Test.PubTator",
    ]),
    ("cdr",[
        "cdr/CDR_TrainingSet.PubTator.txt",
        "cdr/CDR_DevelopmentSet.PubTator.txt",
        "cdr/CDR_TestSet.PubTator.txt",
    ]),
    ("aimed",["aimed/aimed_bioc.Sen.PubTator"]),
    ("hprd50",["hprd50/hprd50_bioc.PubTator"]),
    ("disgenet",["disgenet/DisGeNET.PubTator"]),
    ("emu_bc",["emu_bc/BCa_db_novel_128.PubTator"]),
    ("emu_pc",   ["emu_pc/PCa_db_novel_51.PubTator"]),
    ("pharmgkb", ["pharmgkb/PharmGKB.PubTator"]),
]

MentionMode = Literal["first_mention", "all_mentions"]

nlp = spacy.blank("en")

def _build_char_to_token(doc):
    c2t = {}
    for i, token in enumerate(doc):
        for ch in range(token.idx, token.idx + len(token.text)):
            c2t[ch] = i
    return c2t


def _resolve_token(c2t, char):
    t = c2t.get(char)
    if t is not None:
        return t
    for delta in (1, -1, 2, -2):
        t = c2t.get(char + delta)
        if t is not None:
            return t
    return None


def _parse_pubtator_file(filepath):
    """Parse a single PubTator file into a list of document dicts.

    Each dict contains the PMID, combined title+abstract text, entity spans,
    concept-to-annotation mappings, and raw relation entries.
    """
    documents = []
    current = None

    def _flush():
        nonlocal current
        if current and current["text"].strip():
            documents.append(current)
        current = None

    with open(filepath, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")

            if line == "":
                _flush()
                continue

            if "|t|" in line:
                _flush()
                parts = line.split("|t|", 1)
                current = {
                    "doc_id":          parts[0],
                    "text":            parts[1],
                    "entities":        {},
                    "concept_to_ann":  {},   # first mention only
                    "concept_to_anns": defaultdict(list),  # all mentions
                    "relations":[],
                }
                continue

            if "|a|" in line and current:
                parts = line.split("|a|", 1)
                abstract = parts[1]
                if abstract:
                    current["text"] = current["text"] + "\n" + abstract
                continue

            if current is None:
                continue

            parts = line.split("\t")

            # Entity line: PMID \t start \t end \t text \t type [\t id]
            if len(parts) >= 5 and parts[1].isdigit():
                start       = int(parts[1])
                end         = int(parts[2])
                entity_text = parts[3]
                entity_type = parts[4]
                # Allow missing concept IDs
                concept_id  = parts[5] if len(parts) >= 6 else ""

                if entity_type not in ENTITY_TYPE_MAP:
                    continue

                # Use E instead of T to avoid collision with AIMed/HPRD native IDs
                ann_id = f"E{len(current['entities'])}"
                current["entities"][ann_id] = {
                    "start":      start,
                    "end":        end,
                    "text":       entity_text,
                    "type":       entity_type,
                    "concept_id": concept_id,
                }

                if concept_id:
                    # Capture ALL secondary concept IDs (split by ; and ,)
                    for cid_group in concept_id.split(";"):
                        for cid in cid_group.split(","):
                            cid = cid.strip()
                            if not cid:
                                continue
                            # First-mention index (original behaviour)
                            if cid not in current["concept_to_ann"]:
                                current["concept_to_ann"][cid] = ann_id
                            # All-mentions index (new)
                            current["concept_to_anns"][cid].append(ann_id)

            # Relation line: PMID \t rel_type \t arg1 \t arg2
            elif len(parts) >= 4 and not parts[1].isdigit():
                rel_type = parts[1]
                arg1_raw = parts[2]
                arg2_raw = parts[3]
                if rel_type not in RELATION_TYPE_MAP:
                    continue
                current["relations"].append({
                    "type":     rel_type,
                    "arg1_raw": arg1_raw,
                    "arg2_raw": arg2_raw,
                })

    _flush()
    return documents


def _resolve_relation_arg_first(doc, arg_raw):
    if arg_raw in doc["entities"]:
        return arg_raw
    if arg_raw in doc["concept_to_ann"]:
        return doc["concept_to_ann"][arg_raw]
    # Fuzzy match for variant IDs
    for cid, ann_id in doc["concept_to_ann"].items():
        if arg_raw in cid or cid in arg_raw:
            return ann_id
    return None


def _resolve_relation_arg_all(doc, arg_raw):
    if arg_raw in doc["entities"]:
        return [arg_raw]
    if arg_raw in doc["concept_to_anns"]:
        return list(doc["concept_to_anns"][arg_raw])
    # Fuzzy match for variant IDs
    matches = []
    for cid, ann_ids in doc["concept_to_anns"].items():
        if arg_raw in cid or cid in arg_raw:
            matches.extend(ann_ids)
    return matches if matches else[]


def _doc_to_gliner(
    doc,
    mention_mode: MentionMode = "first_mention",
    max_mentions_per_triple: int = 3,
):
    """Convert a parsed PubTator document dict into a GLiNER training sample.

    Tokenises the text, maps char offsets to token indices, and builds NER and
    relation lists according to the chosen mention_mode. Returns None if the
    document has no mappable entity spans.
    """
    text = doc["text"]
    if not text or len(text) < 5:
        return None

    spacy_doc = nlp(text)
    tokens    = [t.text for t in spacy_doc]
    c2t       = _build_char_to_token(spacy_doc)

    if len(tokens) < 3:
        return None

    # Build NER list and ann_id → ner_idx map
    ner_list       =[]
    ann_to_ner_idx = {}

    for ann_id, ent in doc["entities"].items():
        start_tok = _resolve_token(c2t, ent["start"])
        end_tok   = _resolve_token(c2t, ent["end"] - 1)
        if start_tok is None or end_tok is None:
            continue
        
        # Enforce valid span ranges to avoid dataloader crashes
        if start_tok > end_tok:
            start_tok, end_tok = end_tok, start_tok
            
        mapped_type              = ENTITY_TYPE_MAP.get(ent["type"], ent["type"])
        ann_to_ner_idx[ann_id]   = len(ner_list)
        ner_list.append([start_tok, end_tok, mapped_type])

    if not ner_list:
        return None

    # Build a span-key → ner_idx map for same-span guard
    span_to_idx = {
        (row[0], row[1]): idx for idx, row in enumerate(ner_list)
    }

    relations_list =[]
    seen_rels      = set()

    if mention_mode == "first_mention":
        for rel in doc["relations"]:
            arg1_ann = _resolve_relation_arg_first(doc, rel["arg1_raw"])
            arg2_ann = _resolve_relation_arg_first(doc, rel["arg2_raw"])
            if arg1_ann is None or arg2_ann is None:
                continue
            if arg1_ann not in ann_to_ner_idx or arg2_ann not in ann_to_ner_idx:
                continue
            mapped_rel = RELATION_TYPE_MAP.get(rel["type"], rel["type"])
            key        = (ann_to_ner_idx[arg1_ann], ann_to_ner_idx[arg2_ann], mapped_rel)
            if key not in seen_rels:
                relations_list.append(list(key))
                seen_rels.add(key)

    else:
        triple_counts: dict[tuple, int] = defaultdict(int)

        ann_to_concept = {}
        for ann_id, ent in doc["entities"].items():
            c_id_raw = ent.get("concept_id", "")
            if c_id_raw:
                primary_cid = c_id_raw.split(";")[0].split(",")[0].strip()
            else:
                primary_cid = ann_id
            ann_to_concept[ann_id] = primary_cid

        for rel in doc["relations"]:
            head_anns  = _resolve_relation_arg_all(doc, rel["arg1_raw"])
            tail_anns  = _resolve_relation_arg_all(doc, rel["arg2_raw"])
            mapped_rel = RELATION_TYPE_MAP.get(rel["type"], rel["type"])

            random.shuffle(head_anns)
            random.shuffle(tail_anns)

            for h_ann in head_anns:
                if h_ann not in ann_to_ner_idx:
                    continue
                h_idx  = ann_to_ner_idx[h_ann]
                h_span = (ner_list[h_idx][0], ner_list[h_idx][1])

                for t_ann in tail_anns:
                    if t_ann not in ann_to_ner_idx:
                        continue
                    t_idx  = ann_to_ner_idx[t_ann]
                    t_span = (ner_list[t_idx][0], ner_list[t_idx][1])

                    # Same-span guard
                    if h_span == t_span:
                        continue

                    # Deduplication
                    dup_key = (h_idx, t_idx, mapped_rel)
                    if dup_key in seen_rels:
                        continue

                    # Per-triple cap
                    h_cid   = ann_to_concept.get(h_ann, h_ann)
                    t_cid   = ann_to_concept.get(t_ann, t_ann)
                    cap_key = (h_cid, t_cid, mapped_rel)
                    if triple_counts[cap_key] >= max_mentions_per_triple:
                        continue

                    relations_list.append(list(dup_key))
                    seen_rels.add(dup_key)
                    triple_counts[cap_key] += 1

    return {
        "tokenized_text": tokens,
        "ner":            ner_list,
        "relations":      relations_list,
    }


def _parse_drugprot(biorex_dir):
    """Parse DrugProt from its separate abstracts/entities/relations TSVs."""
    base_dirs =[
        os.path.join(biorex_dir, "drugprot", "drugprot-gs-training-development", "training"),
        os.path.join(biorex_dir, "drugprot", "drugprot-gs-training-development"),
    ]

    documents = {}

    for base in base_dirs:
        abs_file = ent_file = rel_file = None
        for f in glob.glob(os.path.join(base, "*.tsv")):
            fname = os.path.basename(f).lower()
            if "abstracs" in fname or "abstract" in fname:
                abs_file = f
            elif "entities" in fname:
                ent_file = f
            elif "relations" in fname:
                rel_file = f

        if not all([abs_file, ent_file]):
            continue

        print(f"    {os.path.relpath(base, biorex_dir)}/")

        with open(abs_file, encoding="utf-8") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t", 2)
                if len(parts) >= 2:
                    pmid     = parts[0]
                    title    = parts[1]
                    abstract = parts[2] if len(parts) > 2 else ""
                    documents[pmid] = {
                        "doc_id":          pmid,
                        "text":            title + "\n" + abstract if abstract else title,
                        "entities":        {},
                        "concept_to_ann":  {},
                        "concept_to_anns": defaultdict(list),
                        "relations":[],
                    }

        # Entities: PMID \t TID \t type \t start \t end \t text
        with open(ent_file, encoding="utf-8") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 6:
                    continue
                pmid, tid, etype      = parts[0], parts[1], parts[2]
                start, end, etext     = parts[3], parts[4], parts[5]
                if pmid not in documents or etype not in ENTITY_TYPE_MAP:
                    continue
                documents[pmid]["entities"][tid] = {
                    "start":      int(start),
                    "end":        int(end),
                    "text":       etext,
                    "type":       etype,
                    "concept_id": tid,
                }
                # DrugProt uses TID as concept ID (no shared concepts)
                documents[pmid]["concept_to_ann"][tid]  = tid
                documents[pmid]["concept_to_anns"][tid].append(tid)

        # Relations: PMID \t rel_type \t Arg1:TID \t Arg2:TID
        if rel_file:
            with open(rel_file, encoding="utf-8") as fh:
                for line in fh:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 4:
                        continue
                    pmid     = parts[0]
                    rel_type = parts[1]
                    arg1     = parts[2].replace("Arg1:", "").replace("Arg2:", "")
                    arg2     = parts[3].replace("Arg1:", "").replace("Arg2:", "")
                    if pmid not in documents or rel_type not in RELATION_TYPE_MAP:
                        continue
                    documents[pmid]["relations"].append({
                        "type":     rel_type,
                        "arg1_raw": arg1,
                        "arg2_raw": arg2,
                    })

    results =[]
    for doc in documents.values():
        sample = _doc_to_gliner(doc)
        if sample is not None:
            results.append(sample)
    return results


def _count_missed_relations(biorex_dir: str = BIOREX_DIR) -> dict:
    """Compare first_mention vs all_mentions relation counts across all docs."""
    print("\n" + "=" * 60)
    print("Missed-data analysis: first_mention vs all_mentions")
    print("=" * 60)

    summary = {}
    total_first = total_all = total_docs = 0

    for dataset_name, file_paths in PUBTATOR_SOURCES:
        ds_first = ds_all = ds_docs = 0

        for rel_path in file_paths:
            full_path = os.path.join(biorex_dir, rel_path)
            if not os.path.exists(full_path):
                continue

            documents = _parse_pubtator_file(full_path)
            for doc in documents:
                s_first = _doc_to_gliner(doc, mention_mode="first_mention")
                s_all   = _doc_to_gliner(doc, mention_mode="all_mentions")
                if s_first is None or s_all is None:
                    continue
                ds_docs  += 1
                ds_first += len(s_first["relations"])
                ds_all   += len(s_all["relations"])

        if ds_docs == 0:
            continue

        gained    = ds_all - ds_first
        pct       = (gained / ds_first * 100) if ds_first > 0 else float("inf")
        summary[dataset_name] = {
            "docs":        ds_docs,
            "first":       ds_first,
            "all":         ds_all,
            "gained":      gained,
            "gained_pct":  pct,
        }
        total_first += ds_first
        total_all   += ds_all
        total_docs  += ds_docs

    col = 22
    print(f"\n  {'Dataset':<{col}} {'Docs':>6} {'first':>8} {'all':>8} {'gained':>8} {'%gain':>7}")
    print(f"  {'-' * (col + 40)}")
    for ds, st in summary.items():
        print(
            f"  {ds:<{col}} {st['docs']:>6} {st['first']:>8} "
            f"{st['all']:>8} {st['gained']:>8} {st['gained_pct']:>6.1f}%"
        )
    print(f"  {'-' * (col + 40)}")
    total_gained = total_all - total_first
    total_pct    = (total_gained / total_first * 100) if total_first > 0 else float("inf")
    print(
        f"  {'TOTAL':<{col}} {total_docs:>6} {total_first:>8} "
        f"{total_all:>8} {total_gained:>8} {total_pct:>6.1f}%"
    )
    print()

    summary["_totals"] = {
        "docs":       total_docs,
        "first":      total_first,
        "all":        total_all,
        "gained":     total_gained,
        "gained_pct": total_pct,
    }
    return summary


def get_biorex_data(
    biorex_dir: str = BIOREX_DIR,
    mention_mode: MentionMode = "all_mentions",
    max_mentions_per_triple: int = 3,
) -> list:
    """Parse all BioREx PubTator source files into GLiNER format."""
    print(f"\nMention mode: {mention_mode!r}  "
          f"(max_mentions_per_triple={max_mentions_per_triple})")

    all_samples    =[]
    dataset_stats  = {}

    for dataset_name, file_paths in PUBTATOR_SOURCES:
        dataset_samples =[]

        for rel_path in file_paths:
            full_path = os.path.join(biorex_dir, rel_path)
            if not os.path.exists(full_path):
                print(f"  WARNING: {full_path} not found, skipping.")
                continue

            print(f"  Parsing {rel_path}...")
            documents = _parse_pubtator_file(full_path)
            print(f"    {len(documents)} documents")

            converted = unresolved = 0
            for doc in documents:
                # Count unresolved args using first_mention resolver as baseline
                for rel in doc["relations"]:
                    a1 = _resolve_relation_arg_first(doc, rel["arg1_raw"])
                    a2 = _resolve_relation_arg_first(doc, rel["arg2_raw"])
                    if a1 is None or a2 is None:
                        unresolved += 1
                sample = _doc_to_gliner(
                    doc,
                    mention_mode=mention_mode,
                    max_mentions_per_triple=max_mentions_per_triple,
                )
                if sample is not None:
                    dataset_samples.append(sample)
                    converted += 1

            if unresolved:
                print(f"    {unresolved} unresolvable relation args")
            print(f"    -> {converted} GLiNER samples")

        n_pos = sum(1 for s in dataset_samples if s["relations"])
        n_neg = len(dataset_samples) - n_pos
        dataset_stats[dataset_name] = {
            "total": len(dataset_samples),
            "pos":   n_pos,
            "neg":   n_neg,
        }
        all_samples.extend(dataset_samples)

    # DrugProt
    print(f"\n  Parsing DrugProt...")
    dp_samples = _parse_drugprot(biorex_dir)
    n_pos = sum(1 for s in dp_samples if s["relations"])
    n_neg = len(dp_samples) - n_pos
    dataset_stats["drugprot"] = {"total": len(dp_samples), "pos": n_pos, "neg": n_neg}
    print(f"    -> {len(dp_samples)} GLiNER samples")
    all_samples.extend(dp_samples)

    # Summary table
    print(f"\n{'='*55}")
    print(f"  {'Dataset':<20s} {'Total':>7} {'Pos':>7} {'Neg':>7}")
    print(f"  {'-'*45}")
    for ds, st in dataset_stats.items():
        print(f"  {ds:<20s} {st['total']:>7} {st['pos']:>7} {st['neg']:>7}")
    print(f"  {'-'*45}")
    total_pos = sum(s["pos"] for s in dataset_stats.values())
    total_neg = sum(s["neg"] for s in dataset_stats.values())
    print(f"  {'TOTAL':<20s} {len(all_samples):>7} {total_pos:>7} {total_neg:>7}")

    ent_counts = Counter()
    rel_counts = Counter()
    for s in all_samples:
        for _, _, label in s["ner"]:
            ent_counts[label] += 1
        for _, _, rtype in s["relations"]:
            rel_counts[rtype] += 1

    print(f"\n  Entity distribution:")
    for label, cnt in ent_counts.most_common():
        print(f"    {label:<25s} {cnt:>8}")
    print(f"\n  Relation distribution:")
    for label, cnt in rel_counts.most_common():
        print(f"    {label:<30s} {cnt:>8}")

    return all_samples
