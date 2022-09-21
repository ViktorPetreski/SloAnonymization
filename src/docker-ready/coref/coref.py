from fastapi import Body, FastAPI
from pydantic import BaseModel
from CorefResolver import CorefResolver
import  os
import gdown
import tarfile
import shutil


app = FastAPI(
    title="Coref REST API",
    description="Returns coreference resolved string"
)

class _CorefRequest(BaseModel):
    text: str


coref_model = {}

@app.on_event("startup")
async def startup_event():
    gdown.download(id="16vz0K7Xd_3nujBC8AxNKOLBCCa9qOikj", output="sloberta.tar.gz")
    file = tarfile.open("./sloberta.tar.gz")
    file.extractall("./sloberta-coref")
    file.close()
    coref_model["model"] = CorefResolver("./sloberta-coref")
    shutil.rmtree("./sloberta.tar.gz")
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
                text='Predsedniški kandidat Levice bo Miha Kordiš'
            ),
            default=None,
            media_type='application/json'
        )
):

    resolved = coref_model["model"].resolve_allennlp(req_body.text)
    return {
        "text": resolved
    }
