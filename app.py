from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from scipy.signal import savgol_filter

app = FastAPI(title="RailSight Proc Python")

# CORS para permitir chamadas do front
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check (para testar no navegador)
@app.get("/health")
def health():
    return {"status": "ok"}

# Estrutura dos dados (se precisar enviar/validar payload no futuro)
class RailData(BaseModel):
    curvature: list[float]
    crosslevel: list[float]
    gauge: list[float]

# Endpoint de processamento com suavização Savitzky–Golay
@app.get("/process")
def process(
    window_m: int = Query(200, description="Tamanho da janela em metros"),
    step_m: int = Query(1, description="Passo em metros"),
):
    # mock de dados brutos (em produção você vai ler dados reais)
    x = np.linspace(0, window_m, num=max(2, window_m // step_m))
    curvature = np.sin(x / 20) + np.random.normal(0, 0.05, len(x))
    crosslevel = np.cos(x / 25) + np.random.normal(0, 0.05, len(x))
    gauge = 1600 + np.random.normal(0, 2, len(x))

    # janela do Savitzky-Golay precisa ser ímpar e <= len(x)
    def sg(y, win=11, poly=3):
        w = min(win, len(y) - (1 - len(y) % 2))  # força ímpar e <= len
        if w < 5:  # sequência muito curta: retorna sem suavizar
            return y.tolist()
        if w % 2 == 0:
            w -= 1
        return savgol_filter(y, w, poly).tolist()

    return {
        "window_m": window_m,
        "step_m": step_m,
        "smoothed": {
            "curvature": sg(curvature),
            "crosslevel": sg(crosslevel),
            "gauge": sg(gauge),
        }
    }
