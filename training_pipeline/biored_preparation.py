"""
Parses the BioRED corpus from BioC XML into GLiNER format.
Supports first_mention and all_mentions coreference resolution modes.
"""

import os
import random
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Literal

import spacy

BIORED_RAW_DIR   = os.path.join("data", "biored_raw")
BIORED_XML_FILES = ["Train.BioC.XML", "Dev.BioC.XML", "Test.BioC.XML"]

ENTITY_LABEL_MAP = {
    "GeneOrGeneProduct":          "Gene",
    "DiseaseOrPhenotypicFeature": "Disease",
    "ChemicalEntity":             "Chemical",
    "SequenceVariant":            "Variant",
    "OrganismTaxon":              "Species",
    "CellLine":                   "Cell Line",
}

# To be fetched by the callback and other modules
RELATION_LABELS = [
    "Association",
    "Positive Correlation",
    "Negative Correlation",
    "Bind",
    "Cotreatment",
    "Comparison",
    "Conversion",
    "Drug Interaction",
]

RELATION_LABEL_MAP = {
    "Association":        "Association",
    "Positive_Correlation": "Positive Correlation",
    "Negative_Correlation": "Negative Correlation",
    "Bind":               "Bind",
    "Cotreatment":        "Cotreatment",
    "Comparison":         "Comparison",
    "Conversion":         "Conversion",
    "Drug_Interaction":   "Drug Interaction",
}

MentionMode = Literal["first_mention", "all_mentions"]

nlp = spacy.blank("en")

_alignment_stats = {"exact": 0, "fallback": 0, "failed": 0}

def _parse_xml_file(xml_path):
    if not os.path.exists(xml_path):
        raise FileNotFoundError(
            f"BioRED XML file not found: {xml_path}\n"
            f"Expected files in {BIORED_RAW_DIR}/: {BIORED_XML_FILES}"
        )
    return ET.parse(xml_path).getroot()


def _get_infon(element, key):
    for infon in element.findall("infon"):
        if infon.get("key") == key:
            return infon.text
    return None


def _reconstruct_document(xml_doc):
    raw_passages = []
    for passage in xml_doc.findall("passage"):
        offset_el = passage.find("offset")
        text_el   = passage.find("text")
        if offset_el is None or text_el is None:
            continue
        raw_passages.append((int(offset_el.text), text_el.text or ""))

    if not raw_passages:
        return ""

    raw_passages.sort(key=lambda x: x[0])
    total_length = max(off + len(txt) for off, txt in raw_passages)
    buf = [" "] * total_length
    for off, txt in raw_passages:
        for i, ch in enumerate(txt):
            buf[off + i] = ch
    return "".join(buf)


def _parse_xml_document(xml_doc):
    """Extract entities and relations from a single BioRED XML document element.

    Rebuilds the full document text from passage offsets, maps entity annotations
    to char spans and concept IDs, and returns raw relation metadata for
    downstream resolution into annotation IDs.
    """
    doc_text = _reconstruct_document(xml_doc)
    doc_id   = xml_doc.findtext("id", default="?")

    entities_meta   = {}
    concept_to_ann  = {}                   # concept_id -> first ann_id
    concept_to_anns = defaultdict(list)    # concept_id -> all ann_ids
    multi_span_cnt  = 0
    dup_id_cnt      = 0

    for passage in xml_doc.findall("passage"):
        for annotation in passage.findall("annotation"):
            ann_id   = annotation.get("id")
            raw_type = _get_infon(annotation, "type")
            label    = ENTITY_LABEL_MAP.get(raw_type)
            if label is None:
                continue

            locations = annotation.findall("location")
            if not locations:
                continue

            starts = []
            ends   = []
            for loc in locations:
                s = int(loc.get("offset"))
                l = int(loc.get("length"))
                starts.append(s)
                ends.append(s + l)

            if len(locations) > 1:
                multi_span_cnt += 1

            span_start = min(starts)
            span_end   = max(ends)

            if ann_id in entities_meta:
                existing = entities_meta[ann_id]
                if (existing["start"], existing["end"], existing["label"]) != (span_start, span_end, label):
                    dup_id_cnt += 1
                    continue

            entities_meta[ann_id] = {
                "start": span_start,
                "end":   span_end,
                "label": label,
            }

            concept_id = _get_infon(annotation, "identifier")
            if concept_id:
                for cid in concept_id.split(","):
                    cid = cid.strip()
                    if cid:
                        if cid not in concept_to_ann:
                            concept_to_ann[cid] = ann_id
                        concept_to_anns[cid].append(ann_id)

    if multi_span_cnt:
        print(f"    [PMID {doc_id}] {multi_span_cnt} multi-span entities (convex-hull used)")
    if dup_id_cnt:
        print(f"    [PMID {doc_id}] {dup_id_cnt} duplicate annotation IDs across passages (first kept)")

    # Store raw concept IDs — resolution into annotation IDs happens in
    # _document_to_sample so it can apply either first_mention or all_mentions.
    relations_meta = []
    unresolved_concepts = 0
    for relation in xml_doc.findall("relation"):
        raw_rel_type = _get_infon(relation, "type")
        rel_type = RELATION_LABEL_MAP.get(raw_rel_type)
        if rel_type not in RELATION_LABELS:
            continue

        arg1_concept = _get_infon(relation, "entity1")
        arg2_concept = _get_infon(relation, "entity2")
        if not (arg1_concept and arg2_concept):
            continue

        # if neither concept resolves to any annotation,
        # count as unresolvable now so the diagnostic still fires.
        if arg1_concept not in concept_to_anns and arg2_concept not in concept_to_anns:
            unresolved_concepts += 1
            continue

        relations_meta.append({
            "type":         rel_type,
            "arg1_concept": arg1_concept,
            "arg2_concept": arg2_concept,
        })

    if unresolved_concepts:
        print(f"    [PMID {doc_id}] {unresolved_concepts} relations with unresolvable concept IDs")

    return doc_text, entities_meta, relations_meta, concept_to_ann, concept_to_anns


def _build_char_to_token(spacy_doc):
    c2t = {}
    for i, token in enumerate(spacy_doc):
        for ch in range(token.idx, token.idx + len(token.text)):
            c2t[ch] = i
    return c2t


def _resolve_token(c2t, char):
    exact = c2t.get(char)
    if exact is not None:
        _alignment_stats["exact"] += 1
        return exact
    # Try ±1 then ±2 as fallback
    for delta in (1, -1, 2, -2):
        fb = c2t.get(char + delta)
        if fb is not None:
            _alignment_stats["fallback"] += 1
            return fb
    _alignment_stats["failed"] += 1
    return None


def _document_to_sample(
    doc_text,
    entities_meta,
    relations_meta,
    concept_to_ann,
    concept_to_anns,
    mention_mode: MentionMode = "all_mentions",
    max_mentions_per_triple: int = 3,
):
    """Convert parsed BioRED document data into a GLiNER training sample.

    Tokenises the document, maps char spans to token indices, and builds NER
    and relation lists according to the chosen mention_mode. Returns None if
    no entity spans could be aligned to tokens.
    """
    spacy_doc = nlp(doc_text)
    tokens    = [t.text for t in spacy_doc]
    c2t       = _build_char_to_token(spacy_doc)

    ner_list       = []
    eid_to_ner_idx = {}

    for eid, emeta in entities_meta.items():
        start_tok = _resolve_token(c2t, emeta["start"])
        end_tok   = _resolve_token(c2t, emeta["end"] - 1)
        if start_tok is None or end_tok is None:
            continue
        # Enforce valid span order
        if start_tok > end_tok:
            start_tok, end_tok = end_tok, start_tok
        eid_to_ner_idx[eid] = len(ner_list)
        ner_list.append([start_tok, end_tok, emeta["label"]])

    if not ner_list:
        return None

    relations_list = []
    seen_rels      = set()

    if mention_mode == "first_mention":
        for rel in relations_meta:
            a1 = concept_to_ann.get(rel["arg1_concept"])
            a2 = concept_to_ann.get(rel["arg2_concept"])
            if a1 is None or a2 is None:
                continue
            if a1 not in eid_to_ner_idx or a2 not in eid_to_ner_idx:
                continue
            h_idx = eid_to_ner_idx[a1]
            t_idx = eid_to_ner_idx[a2]
            # Same-span guard
            if (ner_list[h_idx][0], ner_list[h_idx][1]) == (ner_list[t_idx][0], ner_list[t_idx][1]):
                continue
            key = (h_idx, t_idx, rel["type"])
            if key not in seen_rels:
                relations_list.append(list(key))
                seen_rels.add(key)

    else:
        # all_mentions: expand each concept to all its annotation instances
        triple_counts: dict = defaultdict(int)

        for rel in relations_meta:
            head_anns  = list(concept_to_anns.get(rel["arg1_concept"], []))
            tail_anns  = list(concept_to_anns.get(rel["arg2_concept"], []))
            rtype      = rel["type"]
            cap_key    = (rel["arg1_concept"], rel["arg2_concept"], rtype)

            # Shuffle so the cap doesn't always select the same pair
            random.shuffle(head_anns)
            random.shuffle(tail_anns)

            for h_ann in head_anns:
                if h_ann not in eid_to_ner_idx:
                    continue
                h_idx  = eid_to_ner_idx[h_ann]
                h_span = (ner_list[h_idx][0], ner_list[h_idx][1])

                for t_ann in tail_anns:
                    if t_ann not in eid_to_ner_idx:
                        continue
                    t_idx  = eid_to_ner_idx[t_ann]
                    t_span = (ner_list[t_idx][0], ner_list[t_idx][1])

                    # Same-span guard
                    if h_span == t_span:
                        continue

                    # Deduplication
                    dup_key = (h_idx, t_idx, rtype)
                    if dup_key in seen_rels:
                        continue

                    # Per-triple cap
                    if triple_counts[cap_key] >= max_mentions_per_triple:
                        continue

                    relations_list.append(list(dup_key))
                    seen_rels.add(dup_key)
                    triple_counts[cap_key] += 1

    return {"tokenized_text": tokens, "ner": ner_list, "relations": relations_list}


def get_biored_data(
    mention_mode: MentionMode = "all_mentions",
    max_mentions_per_triple: int = 3,
) -> list:
    """Parse all BioRED XML files into a list of GLiNER training samples.

    Args:
        mention_mode: "first_mention" or "all_mentions" coreference strategy.
        max_mentions_per_triple: Maximum relation instances generated per concept pair.

    Returns a list of GLiNER-format samples.
    """
    global _alignment_stats
    _alignment_stats = {"exact": 0, "fallback": 0, "failed": 0}

    print(f"\nMention mode: {mention_mode!r}  "
          f"(max_mentions_per_triple={max_mentions_per_triple})")

    formatted_data = []
    total_docs     = 0
    skipped_docs   = 0

    for xml_filename in BIORED_XML_FILES:
        xml_path = os.path.join(BIORED_RAW_DIR, xml_filename)
        print(f"Parsing {xml_path} ...")
        try:
            collection = _parse_xml_file(xml_path)
        except FileNotFoundError as e:
            print(f"  WARNING: {e} -- skipping.")
            continue

        documents = collection.findall("document")
        print(f"  {len(documents)} documents found.")
        total_docs += len(documents)

        for xml_doc in documents:
            doc_text, entities_meta, relations_meta, concept_to_ann, concept_to_anns = \
                _parse_xml_document(xml_doc)
            if not entities_meta:
                skipped_docs += 1
                continue
            sample = _document_to_sample(
                doc_text, entities_meta, relations_meta,
                concept_to_ann, concept_to_anns,
                mention_mode, max_mentions_per_triple,
            )
            if sample is not None:
                formatted_data.append(sample)
            else:
                skipped_docs += 1

    total_lookups = sum(_alignment_stats.values())
    if total_lookups:
        print(
            f"\n  Token alignment stats: "
            f"{_alignment_stats['exact']} exact, "
            f"{_alignment_stats['fallback']} fallback (±1/±2), "
            f"{_alignment_stats['failed']} failed "
            f"({_alignment_stats['fallback']/total_lookups*100:.1f}% fallback rate)"
        )

    print(
        f"\nBioRED parsing complete.\n"
        f"  Documents processed : {total_docs - skipped_docs} "
        f"({skipped_docs} skipped — no mappable entities)\n"
        f"  GLiNER samples      : {len(formatted_data)}"
    )
    return formatted_data
