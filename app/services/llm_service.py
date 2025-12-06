import os
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class LLMService:
    """
    Loads a fine-tuned GPT-2 model if available; otherwise falls back to base GPT-2.
    """

    def __init__(
        self,
        weights_dir: str = "weights/gpt2_squad_ft",
        device: str = "cpu",
    ):
        self.device = device

        if os.path.isdir(weights_dir) and os.path.isfile(os.path.join(weights_dir, "config.json")):
            self.tokenizer = AutoTokenizer.from_pretrained(weights_dir)
            self.model = AutoModelForCausalLM.from_pretrained(weights_dir).to(self.device)
            self.is_finetuned = True
            self.weights_dir = weights_dir
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
            self.model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").to(self.device)
            self.is_finetuned = False
            self.weights_dir = None

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

        self.model.eval()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_p: float = 0.9,
    ) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return text
