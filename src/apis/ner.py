from fastapi import Body, FastAPI
from pydantic import BaseModel
from NERPredictor import NERPredictor
import  os

app = FastAPI(
    title="NER REST API",
    description="Finds named entities in text"
)

class _NERRequest(BaseModel):
    text: str

ner_model_path = os.getenv("NER_MODEL_PATH", None)

ner_model = {}

@app.on_event("startup")
async def startup_event():
    ner_model["model"] = NERPredictor("EMBEDDIA/sloberta", ner_model_path)
    return ner_model

@app.get("/livness")
async def vers():
    return {
        "text": "ok"
    }

@app.post("/pos/predict")
async def predict(
        req_body: _NERRequest = Body(
            example=_NERRequest(
                text='Janez Novak je šel v Mercator. Tam je kupil mleko. Nato ga je spreletela misel, da bi moral iti v Hofer.'
            ),
            default=None,
            media_type='application/json'
        )
):
    if ner_model_path is None:
        raise Exception(
            "POS model path not specified. Set environment variable COREF_MODEL_PATH as path to the model to load.")
    model = ner_model["model"]
    resolved = model.predict(req_body.text)
    return {
        "text": resolved
    }
