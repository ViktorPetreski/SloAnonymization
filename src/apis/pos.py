from fastapi import Body, FastAPI
from pydantic import BaseModel
from POSPredictor import POSPredictor
import  os

app = FastAPI(
    title="Coref REST API",
    description="Returns coreference resolved string"
)

class _PosRequest(BaseModel):
    text: str

pos_model_path = os.getenv("POS_MODEL_PATH", None)

pos_model = {}

@app.on_event("startup")
async def startup_event():
    pos_model["model"] = POSPredictor("EMBEDDIA/sloberta", pos_model_path)
    return pos_model

@app.get("/livness")
async def vers():
    return {
        "text": "ok"
    }

@app.post("/pos/predict")
async def predict(
        req_body: _PosRequest = Body(
            example=_PosRequest(
                text='Janez Novak je šel v Mercator. Tam je kupil mleko. Nato ga je spreletela misel, da bi moral iti v Hofer.'
            ),
            default=None,
            media_type='application/json'
        )
):
    if pos_model_path is None:
        raise Exception(
            "POS model path not specified. Set environment variable COREF_MODEL_PATH as path to the model to load.")
    model = pos_model["model"]
    resolved = model.predict(req_body.text)
    return {
        "text": resolved
    }
