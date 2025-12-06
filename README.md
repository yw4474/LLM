# GenAI API (CNN + Fine-tuned GPT-2)

This repository contains a FastAPI project that integrates:
1) A CNN image classifier (for CIFAR-10 style inference by image path).
2) A fine-tuned GPT-2 LLM on the SQuAD dataset.
3) A new API endpoint `/generate_with_llm` added based on Module 9 requirements.

The project is Dockerized so the instructor can build and run the server on their machine.

---

## Project Structure

```text
.
├── app/
│   ├── main.py
│   ├── api/
│   │   └── router.py
│   ├── routers/
│   │   └── text_generation.py
│   ├── services/
│   │   └── llm_service.py
│   └── image_classifier.py
├── scripts/
│   └── train_gpt2_squad.py
├── weights/
│   └── gpt2_squad_ft/   (generated after training)
├── requirements.txt
├── Dockerfile
├── pyproject.toml
└── README.md
