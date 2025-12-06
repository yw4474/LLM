import os
import random
from dataclasses import dataclass
from typing import List, Dict

import torch
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)

from torch.optim import AdamW

# -----------------------------
# 1) Your required "answer style"
# -----------------------------
START_PHRASE = "That is a great question. "
END_PHRASE = " Let me know if you have any other questions."

# You can change these two lines to any format you want.
# The assignment explicitly says you can choose the format. :contentReference[oaicite:4]{index=4}


# -----------------------------
# 2) Lightweight dataset wrapper
# -----------------------------
class CausalTextDataset(Dataset):
    def __init__(self, encodings: Dict[str, torch.Tensor]):
        self.encodings = encodings

    def __len__(self):
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        # For causal LM, labels are the same as input_ids
        item["labels"] = item["input_ids"].clone()
        return item


# -----------------------------
# 3) Build training texts from SQuAD
# -----------------------------
def build_texts_from_squad(split, max_samples: int = 2000) -> List[str]:
    """
    Convert SQuAD examples into a single string prompt that GPT-2 can learn.
    We include the answer in the training text so the model learns the style.
    """
    texts = []
    n = min(len(split), max_samples)

    for i in range(n):
        ex = split[i]
        question = ex["question"].strip()
        context = ex["context"].strip()

        # SQuAD answers is a dict with "text" list.
        answer_list = ex["answers"]["text"]
        answer = answer_list[0].strip() if len(answer_list) > 0 else ""

        # Training format (simple + consistent)
        # The model learns that answers must include prefix/suffix.
        formatted_answer = f"{START_PHRASE}{answer}{END_PHRASE}"

        text = (
            "Question: " + question + "\n"
            "Context: " + context + "\n"
            "Answer: " + formatted_answer
        )
        texts.append(text)

    return texts


# -----------------------------
# 4) Tokenization helper
# -----------------------------
def tokenize_texts(tokenizer, texts: List[str], max_length: int = 256):
    return tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )


# -----------------------------
# 5) Training loop
# -----------------------------
def train(
    model,
    train_loader,
    device="cpu",
    lr=5e-5,
    epochs=1,
):
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr)

    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(10, total_steps // 10),
        num_training_steps=total_steps,
    )

    step = 0
    for ep in range(epochs):
        running = 0.0
        for batch in train_loader:
            step += 1
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            running += loss.item()
            if step % 50 == 0:
                avg = running / 50
                print(f"[train] epoch={ep+1} step={step} loss={avg:.4f}")
                running = 0.0


# -----------------------------
# 6) Save fine-tuned model
# -----------------------------
def save_model_and_tokenizer(model, tokenizer, out_dir="weights/gpt2_squad_ft"):
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[ok] saved model to: {out_dir}")


def main():
    # Reproducibility
    random.seed(42)
    torch.manual_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load SQuAD dataset (assignment requirement) :contentReference[oaicite:5]{index=5}
    ds = load_dataset("squad")

    # Prepare texts
    train_texts = build_texts_from_squad(ds["train"], max_samples=2000)
    test_texts = build_texts_from_squad(ds["validation"], max_samples=300)

    # Load base GPT-2 (module 9 requirement) :contentReference[oaicite:6]{index=6}
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").to(device)

    # GPT-2 has no pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    # Tokenize
    train_enc = tokenize_texts(tokenizer, train_texts)
    test_enc = tokenize_texts(tokenizer, test_texts)

    train_dataset = CausalTextDataset(train_enc)
    test_dataset = CausalTextDataset(test_enc)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    _ = DataLoader(test_dataset, batch_size=2, shuffle=False)

    # Train (lightweight; you can increase epochs if you want)
    train(model, train_loader, device=device, epochs=1)

    # Save
    save_model_and_tokenizer(model, tokenizer)


if __name__ == "__main__":
    main()
