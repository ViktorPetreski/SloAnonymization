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

ner_model = {}

@app.on_event("startup")
async def startup_event():
    ner_model["model"] = NERPredictor("EMBEDDIA/sloberta", "vpetreski/sloberta-NER")
    ner_model["model"].init_all()
    return ner_model

@app.get("/livness")
async def vers():
    return {
        "text": "ok"
    }

@app.post("/ner/predict")
async def predict(
        req_body: _NERRequest = Body(
            example=_NERRequest(
                text='Predsedniški kandidat Levice bo Miha Kordiš'
            ),
            default=None,
            media_type='application/json'
        )
):

    model = ner_model["model"]
    resolved = model.predict(req_body.text)
    return {
        "text": resolved,
        "tag_dist": model.tag_dist,
        "word_to_tag": model.word_to_tag
    }
