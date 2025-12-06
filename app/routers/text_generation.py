from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.services.llm_service import LLMService


router = APIRouter()

# Align with Module 9 style of request object. :contentReference[oaicite:8]{index=8}
class TextGenerationRequest(BaseModel):
    # Keep compatibility with the class activity naming
    start_word: Optional[str] = Field(default=None, description="Legacy prompt field")
    length: Optional[int] = Field(default=None, description="Legacy length field")

    # New clearer fields
    prompt: Optional[str] = Field(default=None, description="Prompt for LLM generation")
    max_new_tokens: int = Field(default=50, ge=1, le=200)

    # Optional QA-style support
    question: Optional[str] = None
    context: Optional[str] = None


# Instantiate once (CPU default for portability)
llm = LLMService(device="cpu")


@router.post("/generate_with_llm")
def generate_with_llm(request: TextGenerationRequest):
    """
    Generates text using fine-tuned GPT-2 if weights exist.
    """

    # Priority 1: explicit QA
    if request.question:
        if request.context:
            prompt = f"Question: {request.question}\nContext: {request.context}\nAnswer:"
        else:
            prompt = f"Question: {request.question}\nAnswer:"
        out = llm.generate(prompt, max_new_tokens=request.max_new_tokens)
        return {
            "generated_text": out,
            "model_finetuned": llm.is_finetuned,
        }

    # Priority 2: new prompt field
    if request.prompt:
        out = llm.generate(request.prompt, max_new_tokens=request.max_new_tokens)
        return {
            "generated_text": out,
            "model_finetuned": llm.is_finetuned,
        }

    # Priority 3: legacy fields from earlier modules
    if request.start_word:
        max_new = request.length if request.length else request.max_new_tokens
        out = llm.generate(request.start_word, max_new_tokens=max_new)
        return {
            "generated_text": out,
            "model_finetuned": llm.is_finetuned,
        }

    return {
        "generated_text": "",
        "model_finetuned": llm.is_finetuned,
        "warning": "No prompt/question provided.",
    }
