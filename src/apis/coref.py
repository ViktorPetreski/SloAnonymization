from fastapi import Body, FastAPI
from pydantic import BaseModel
from CorefResolver import CorefResolver
import  os

app = FastAPI(
    title="Coref REST API",
    description="Returns coreference resolved string"
)

class _CorefRequest(BaseModel):
    text: str

coref_model_path = os.getenv("COREF_MODEL_PATH", None)
coref_model = {}

@app.on_event("startup")
async def startup_event():
    coref_model["model"] = CorefResolver(coref_model_path)
    return coref_model

@app.get("/livness")
async def vers():
    return {
        "text": "ok"
    }

@app.post("/coref/resolve")
async def predict(
        req_body: _CorefRequest = Body(
            example=_CorefRequest(
                text='Janez Novak je šel v Mercator. Tam je kupil mleko. Nato ga je spreletela misel, da bi moral iti v Hofer.'
            ),
            default=None,
            media_type='application/json'
        )
):
    if coref_model_path is None:
        raise Exception(
            "Coref model path not specified. Set environment variable COREF_MODEL_PATH as path to the model to load.")
    resolved = coref_model["model"].resolve_allennlp(req_body.text)
    return {
        "text": resolved
    }
