# app/main.py

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from app.data_loader import ler_dados
from app.model import treinar_modelo
from app.config import DATA_PATH

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "API está funcionando!"}


@app.get("/treinar/")
async def treinar():
    try:
        df = ler_dados(DATA_PATH)
        modelo, X_test, y_test = treinar_modelo(df)
        return JSONResponse(content={"mensagem": "Modelo treinado com sucesso!"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"erro": str(e)})
