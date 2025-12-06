# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.image_classifier import SimpleImageClassifier
from app.api.router import router as api_router
from app.routers.text_generation import router as text_router


# ---------
# Request schema for classifier
# ---------
class ClassifyRequest(BaseModel):
    image_path: str  # e.g., "sample.jpg" in your project folder


def create_app() -> FastAPI:
    app = FastAPI(title="GenAI API")

    # ---------
    # Include existing routers (GAN/Diffusion/EBM/etc.)
    # ---------
    app.include_router(api_router)

    # ---------
    # Include new LLM router
    # ---------
    app.include_router(text_router)

    # ---------
    # Instantiate classifier once (CPU is fine)
    # ---------
    classifier = SimpleImageClassifier(dataset="cifar10", device="cpu")

    # ---------
    # Simple root + health
    # ---------
    @app.get("/")
    def read_root():
        return {"status": "ok", "message": "GenAI API is running"}

    @app.get("/health")
    def health():
        return {"status": "ok"}

    # ---------
    # CNN classify endpoint
    # ---------
    @app.post("/classify")
    def classify_image(req: ClassifyRequest):
        try:
            result = classifier.predict(req.image_path)
            return {"prediction": result}
        except FileNotFoundError:
            raise HTTPException(status_code=400, detail="Image file not found")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


app = create_app()
