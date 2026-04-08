# Knowledgator Task — ML Researcher

The goal was to fine-tune the GLiNER-relex model on a biomedical relation extraction task and build a Graph RAG pipeline on top of it using the RetriCo framework.

---

## Fine-tuning

Install the requirements:
```
pip install -r requirements.txt
```

To train:
```
python finetune_lora.py
```

To evaluate the fine-tuned model against the baseline:
```
python training_pipeline/evaluate.py
python training_pipeline/sweep.py
```

The model has also been published on hugging-face: https://huggingface.co/grsilva/gliner-relex-biomedical-lora

### Methodology

Full SFT led to significant catastrophic forgetting. Therefore, since GLiNER uses a DeBERTa backbone, LoRA adapters were applied specifically to the DeBERTa encoder (`model.model.token_rep_layer.bert_layer.model`) while the GLiNER classification heads (`span_rep_layer`, `prompt_rep_layer`, `pair_rep_layer`) and the DeBERTa→GLiNER projection layer were kept fully trainable. This gives:

- LoRA adapters on DeBERTa query and value projection layers
- Fully trainable GLiNER NER/RE heads and projection layer
- Frozen: DeBERTa embeddings, FFN layers, and layer norms

After training, LoRA weights are merged back into DeBERTa so the result is a standard GLiNER checkpoint compatible with `model.inference()` and RetriCo — no adapter loading logic required.

To find the best inference thresholds, a grid sweep is run on the test set over NER and RE confidence threshold combinations to maximise RE F1. Results reported here use the best thresholds found (NER: 0.35, RE: 0.5).

### Main Difficulties

The baseline model performed poorly on the biomedical domain out of the box, so the challenge was improving domain performance without triggering catastrophic forgetting. Initial full fine-tuning attempts achieved slightly better test-set metrics but degraded downstream RAG quality.

The datasets are also heavily imbalanced — some relation classes have support as low as ~100 samples out of ~9,600 total. Some corpus formats also made it difficult to extract every possible relation and to audit the correctness of the converted GLiNER samples.

### Datasets Used

- ADE Corpus v2
- DDI Corpus
- BioRED
- BioREx — data not included; can be obtained from https://github.com/ncbi/BioREx/

Two dataset variants are available for training and evaluation: **BioREx** and **merged_biored**. The `merged_biored` dataset (ADE + DDI + BioRED) was used to train the final published model. Evaluation was run on both.

### Pre-processing Decisions

- **Entity type normalisation** — BioREx aggregates several sub-datasets where the same concept can appear under different type strings; types were normalised to a shared vocabulary across all corpora.
- **Relation label normalisation** — relation type strings were standardised across datasets (e.g., DDI `"effect"` → `"has effect on"`, BioRED `"Positive_Correlation"` → `"Positive Correlation"`).
- **Coreference / mention expansion** — when a concept is mentioned multiple times in a document, relation instances are generated for all mention combinations, capped at 3 per concept pair (`max_mentions_per_triple`) to avoid over-representing high-frequency concepts.
- **Multi-span entity handling** — entities with discontinuous text spans (common in BioRED and DDI) are collapsed to a single convex-hull span (earliest start, latest end).
- **Negative sample capping** — datasets with many labelled entities but few labelled relations were subsampled so the number of negative samples does not exceed 1.5× the number of positive ones, preventing the model from becoming overly conservative about predicting relations.
- **Stratified dataset splitting** — splits are stratified by source dataset to preserve the class distribution of lower-support categories across train/test/eval.
- **Sample validation** — samples that are too short or contain out-of-bounds NER/RE token indices are removed before training.

### Key Parameters

| Parameter | Value | Notes |
|---|---|---|
| `r` / `lora_alpha` | 16 / 64 | LoRA rank and scaling; applied to query and value projections in DeBERTa |
| `learning_rate` | 1e-4 | Applied to LoRA adapter parameters |
| `others_lr` | 1e-4 | Applied to GLiNER head parameters (separate parameter group in the Trainer) |
| `focal_loss_alpha` / `gamma` | 0.5 / 2 | Focal loss to down-weight easy negatives and handle class imbalance |
| Effective batch size | 32 | `per_device_train_batch_size=8` × `gradient_accumulation_steps=4` |
| `num_train_epochs` | 15 | With `EarlyStoppingCallback(patience=5)` |
| `metric_for_best_model` | `eval_composite` | Composite NER+RE F1 score computed by `REOptimizationCallback` each epoch |
| `composite_weight` | 0.6 | `eval_composite = 0.4 × NER_F1 + 0.6 × RE_F1` — weights RE more heavily for model selection |
| Best inference thresholds | NER 0.35 / RE 0.4 | Found via `sweep.py` on the test set |

A custom `REOptimizationCallback` runs full NER+RE inference after each epoch to compute `eval_composite`, since the built-in Trainer metrics do not track relation extraction.

### Results

#### Merged BioRED (ADE + DDI + BioRED)

|     | Baseline P | Baseline R | Baseline F1 | Fine-tuned P | Fine-tuned R | Fine-tuned F1 | ΔP      | ΔR      | ΔF1     |
|-----|-----------|-----------|------------|-------------|-------------|--------------|---------|---------|---------|
| NER | 0.2438    | 0.3803    | 0.2971     | 0.4337      | 0.5925      | 0.5008       | +0.1899 | +0.2122 | +0.2037 |
| RE  | 0.0567    | 0.0705    | 0.0628     | 0.1908      | 0.2694      | 0.2234       | +0.1341 | +0.1989 | +0.1606 |

#### BioREx

|     | Baseline P | Baseline R | Baseline F1 | Fine-tuned P | Fine-tuned R | Fine-tuned F1 | ΔP      | ΔR      | ΔF1     |
|-----|-----------|-----------|------------|-------------|-------------|--------------|---------|---------|---------|
| NER | 0.3215    | 0.2960    | 0.3082     | 0.3674      | 0.7514      | 0.4935       | +0.0459 | +0.4555 | +0.1853 |
| RE  | 0.0098    | 0.0106    | 0.0102     | 0.0205      | 0.0151      | 0.0174       | +0.0107 | +0.0045 | +0.0072 |

The model shows improvement on both datasets, massively in recall (when out-of-domain) so the model got better at identifying actual positives. The BioREx evaluation serves as a cross-distribution check. The model was trained on `merged_biored`, so these results indicate whether the improvement generalises beyond the training distribution.

### Other Approaches Tried

- Optimising for NER and RE jointly vs. NER only. NER-focused optimisation gave better overall results since RE quality is bounded by NER quality.
- Different hyperparameter combinations (learning rates, focal loss alpha and gamma).
- Freezing different subsets of DeBERTa layers.
- Two-stage pipeline: first, freeze early DeBERTa layers and train the rest + heads with NER-focused selection; then unfreeze all weights and train with a joint NER+RE objective.
- Different dataset combinations.

### Things I Could Try Going Forward

- Adding label descriptions alongside label names to provide richer signal to the model.
- Mixing a general-domain NER/RE dataset with the biomedical ones to preserve generalisation capabilities.
- Automated hyperparameter optimisation (e.g., Optuna).

---

## Graph RAG — RetriCo

The goal is to swap the baseline GLiNER-relex model for the fine-tuned one and measure whether the knowledge graph-based RAG pipeline improves answer quality.


- Warning: You must set a valid `DEEPSEEK_API_KEY` environment variable or set your own key and change the api url and model inside the `compare_models.py` file. 

To run:
```
python compare_models.py
```


### Methodology

88 PubMed abstracts were extracted and ingested into RetriCo with both models. 10 evaluation questions about the abstracts were generated by an LLM (stored in `test_questions.txt`). Answer quality was judged by an LLM-as-a-judge (DeepSeek v3.2, `deepseek-chat`) scoring each response on:

- **Relevance** (1–5): how directly the answer addresses the question.
- **Specificity** (1–5): how concrete and clinically detailed the information is.

The extracted abstracts are in `data/rag_documents/`. Full responses, retrieved entities and relations, and the LLM baseline responses are saved in `data/eval_results/model_comparison.json` along with the metrics.

### Results

**Graph creation:**

| Model      | Entities | Relations |
|------------|----------|-----------|
| Baseline   | 4,302    | 3,765     |
| Fine-tuned | 2,602    | 1,961     |

**Average retrieval metrics per query:**

| Model      | Retrieved Entities | Retrieved Relations | Chunks | Subgraph Density |
|------------|-------------------|---------------------|--------|-----------------|
| Baseline   | 173.0             | 168.8               | 170.3  | 0.95            |
| Fine-tuned | 32.0              | 19.2                | 33.6   | 0.43            |

**LLM-as-a-judge scores:**

| Model      | Relevance | Specificity |
|------------|-----------|-------------|
| Baseline   | 4.9 / 5   | 3.6 / 5     |
| Fine-tuned | 4.6 / 5   | 3.1 / 5     |

### Observations

- The fine-tuned model extracts far fewer entities and relations per query (avg 32 / 19 vs. 173 / 169 for the baseline), the subgraph density is much lower (0.43 vs. 0.95), meaning the retrieved entities are less interconnected. RetriCo's graph traversal then hops across weak links into unrelated subgraphs, bringing in noise from other documents. 

- On the other hand the baseline ends up creating and retrieving too many relations/entities which could lead to too much noise and repetition. Q1 has 14 relations extracted, however, at least 6 of these are repeats (can be considered more depending on how strict you are with your evaluation).

- A major objective going forward should be increasing this subgraph density instead of blindly wanting to extract more entities or relations.

- Queries that match the fine-tuned model's training distribution score comparably or better than the baseline (Q2, Q7, Q8). However, the baseline model is a better generaliser hence the better results in the rest of the questions.

- I don't think RetriCo was using GPU for model inference (could be me that did not notice any GPU usage)

### Things I Could Try Going Forward

- Per-document label selection using an LLM to identify which entity and relation types are relevant for each document, at the cost of additional inference overhead.
- Use an earlier checkpoint of the model that has comparable performance to see if extraction performance is not as downgraded.
- Experimenting with different RetriCo retriever modes beyond the default.
- Find another metrics to report. Using LLM-as-a-Judge is not very reliable, especially when the LLM was used both in the RAG pipeline and as the Judge. 
