"""
Fine-tunes GLiNER for biomedical NER and relation extraction using LoRA.
Applies LoRA only to the inner DeBERTa encoder while keeping the GLiNER
NER/RE heads fully trainable. After training, merges LoRA weights back
into the base model and pushes the result to HuggingFace Hub.
"""

import os
import json
import glob
import torch

from gliner import GLiNER
from gliner.training import Trainer, TrainingArguments
from gliner.data_processing.collator import RelationExtractionSpanDataCollator

from peft import LoraConfig, get_peft_model

from training_pipeline.data_pipeline import prepare_merged_data
from transformers import EarlyStoppingCallback
from training_pipeline.re_callback import REOptimizationCallback, SafeCollator

os.environ["TOKENIZERS_PARALLELISM"] = "true"

DATASET    = "merged_biored"
# Repo to push
HF_REPO_ID = "grsilva/gliner-relex-biomedical-lora"

splits = prepare_merged_data(mode=DATASET)

train_data = splits["train"]
test_data  = splits["test"]
eval_data  = splits["eval"]

print(
    f"Dataset sizes — "
    f"train: {len(train_data)}, eval: {len(eval_data)}, test: {len(test_data)}"
)

def find_best_checkpoint(checkpoint_dir):
    """Return the best checkpoint path from a training output directory.

    Reads trainer_state.json to find the checkpoint selected by the Trainer
    as best. Falls back to the latest checkpoint by step number if the state
    file is absent or the recorded path no longer exists.
    """
    checkpoints = sorted(
        glob.glob(os.path.join(checkpoint_dir, "checkpoint-*")),
        key=lambda x: int(x.rsplit("-", 1)[-1]),
    )
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints in {checkpoint_dir}/")
    latest     = checkpoints[-1]
    state_path = os.path.join(latest, "trainer_state.json")
    if os.path.exists(state_path):
        with open(state_path) as f:
            state = json.load(f)
        best = state.get("best_model_checkpoint")
        if best and os.path.isdir(best):
            return best
    return latest

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(f"Device: {device}")

model_name = "knowledgator/gliner-relex-large-v1.0"
model      = GLiNER.from_pretrained(model_name)

# Apply LoRA to the inner DeBERTa backbone only, leaving the GLiNER heads untouched.
inner_deberta = model.model.token_rep_layer.bert_layer.model

lora_config = LoraConfig(
    r=16,
    lora_alpha=64,
    target_modules=["query_proj", "value_proj"],
    lora_dropout=0.05,
    bias="none",
)

# Replace the bare DeBERTa model with the PEFT-wrapped version.
peft_deberta = get_peft_model(inner_deberta, lora_config)
model.model.token_rep_layer.bert_layer.model = peft_deberta

# Freeze all parameters first; LoRA adapters and heads are selectively unfrozen below.
for param in model.parameters():
    param.requires_grad = False

# Re-enable gradients for LoRA adapter parameters.
for name, param in model.named_parameters():
    if "lora_" in name:
        param.requires_grad = True

# Unfreeze the GLiNER NER and RE classification heads.
head_modules = [
    "span_rep_layer",
    "prompt_rep_layer",
    "pair_rep_layer",
]

for name, param in model.named_parameters():
    if any(h in name for h in head_modules):
        param.requires_grad = True

# Unfreeze the projection layer that bridges DeBERTa's output to GLiNER's hidden dim.
for name, param in model.named_parameters():
    if "token_rep_layer.projection" in name:
        param.requires_grad = True

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params     = sum(p.numel() for p in model.parameters())
frozen_params    = total_params - trainable_params

print(f"\n  Total params:     {total_params:,}")
print(f"  Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
print(f"  Frozen params:    {frozen_params:,}")

print("\n  Trainable parameter groups:")
groups = {}
for name, param in model.named_parameters():
    if param.requires_grad:
        if "lora_" in name:
            group = "LoRA adapters"
        elif "span_rep_layer" in name:
            group = "span_rep_layer (NER head)"
        elif "prompt_rep_layer" in name:
            group = "prompt_rep_layer"
        elif "pair_rep_layer" in name:
            group = "pair_rep_layer (RE head)"
        elif "projection" in name:
            group = "projection layer"
        else:
            group = "other"
        groups[group] = groups.get(group, 0) + param.numel()

for group, count in sorted(groups.items(), key=lambda x: -x[1]):
    print(f"    {group}: {count:,}")


data_collator = RelationExtractionSpanDataCollator(
    model.config,
    data_processor=model.data_processor,
    prepare_labels=True,
)
safe_collator = SafeCollator(data_collator)

model.to(device)


output_dir = "model_chkpts/gliner_lora_biomedical"
batch_size = 8
accum_steps = 4
epochs = 15

training_args = TrainingArguments(
    output_dir=output_dir,
    remove_unused_columns=False,

    # Training hyperparams
    learning_rate=1e-4,
    weight_decay=0.01,
    others_lr=1e-4,
    others_weight_decay=0.01,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,

    # Loss
    focal_loss_alpha=0.5,
    focal_loss_gamma=2,
    loss_reduction="mean",

    # Batches
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=accum_steps,
    num_train_epochs=epochs,

    # Eval / saving
    eval_strategy="epoch",
    save_strategy="epoch",
    metric_for_best_model="eval_composite",
    greater_is_better=True,
    load_best_model_at_end=True,
    save_total_limit=2,

    # Misc
    dataloader_num_workers=0,
    use_cpu=(not torch.cuda.is_available()),
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=eval_data,
    processing_class=model.data_processor.transformer_tokenizer,
    data_collator=safe_collator,
    callbacks=[
        REOptimizationCallback(
            eval_data,
            dataset=DATASET,
            threshold=0.3,
            relation_threshold=0.4,
            composite_weight=0.6,
            batch_size=12,
        ),
        EarlyStoppingCallback(early_stopping_patience=5),
    ],
)

print("Starting LoRA Fine-Tuning…")
trainer.train()
print(f"Collator dropped {safe_collator.dropped} batches.")

best_ckpt = find_best_checkpoint(output_dir)
print(f"\nBest checkpoint: {best_ckpt}")

print("Re-applying PEFT wrapper to base model for merge...")
model = GLiNER.from_pretrained(model_name)
inner_deberta = model.model.token_rep_layer.bert_layer.model
peft_deberta  = get_peft_model(inner_deberta, lora_config)
model.model.token_rep_layer.bert_layer.model = peft_deberta

# Load checkpoint weights
bin_path = os.path.join(best_ckpt, "pytorch_model.bin")
st_path  = os.path.join(best_ckpt, "model.safetensors")
if os.path.exists(bin_path):
    state_dict = torch.load(bin_path, map_location="cpu")
elif os.path.exists(st_path):
    from safetensors.torch import load_file
    state_dict = load_file(st_path)
else:
    raise FileNotFoundError(f"No model weights found in {best_ckpt}")

result  = model.model.load_state_dict(state_dict, strict=False)
missing = [k for k in result.missing_keys if "lora_" not in k]
if missing:
    print(f"  Non-LoRA missing keys ({len(missing)}): {missing[:5]}")
if result.unexpected_keys:
    print(f"  Unexpected keys ({len(result.unexpected_keys)}): {result.unexpected_keys[:5]}")

model.to(device)

print("Merging LoRA weights into DeBERTa…")

peft_deberta   = model.model.token_rep_layer.bert_layer.model
merged_deberta = peft_deberta.merge_and_unload()
model.model.token_rep_layer.bert_layer.model = merged_deberta

# Save as standard GLiNER checkpoint
merged_output_dir = "gliner_lora_biomedical_merged"
os.makedirs(merged_output_dir, exist_ok=True)
model.save_pretrained(merged_output_dir)
print(f"Merged model saved to {merged_output_dir}/")

print(f"\nPushing to HuggingFace Hub: {HF_REPO_ID}")
try:
    model.push_to_hub(HF_REPO_ID, commit_message="LoRA fine-tuned GLiNER for biomedical RE (merged)")
    print("Push successful!")
except Exception as e:
    print(f"Push failed: {e}")
    print(f"  Manual push: model = GLiNER.from_pretrained('{merged_output_dir}')")
    print(f"  model.push_to_hub('{HF_REPO_ID}')")

print("\nDone.")
