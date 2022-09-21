from fastapi import Body, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from Pipeline import Pipeline

app = FastAPI(
    title="PIPELINE REST API",
    description="Finds named entities in text"
)

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class _NERRequest(BaseModel):
    text: str
    mode: str

pipe = {}

@app.on_event("startup")
async def startup_event():
    pipe["model"] = Pipeline("")
    # ner_model["model"].init_all()
    return pipe

@app.get("/livness")
async def vers():
    return {
        "text": "ok"
    }

@app.post("/annonymize")
async def predict(
        req_body: _NERRequest = Body(
            example=_NERRequest(
                text='Janez Novak je šel v Mercator. Tam je kupil mleko. Nato ga je spreletela misel, da bi moral iti v Hofer.',
                mode=""
            ),
            default=None,
            media_type='application/json'
        )
):
    mode = req_body.mode

    pipe["model"].start_predictions(req_body.text, mode=mode)
    return {
        "text": pipe["model"].text,
    }
