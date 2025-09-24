from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from scipy.signal import savgol_filter
import numpy as np

app = FastAPI(title="RailSight Proc Python", version="1.0.0")

# 🔥 CORS liberado (em produção você pode restringir para seu domínio da Vercel)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],         # ex.: ["https://railsight-landing-general.vercel.app"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DataIn(BaseModel):
    data: list[float] = Field(..., description="Série numérica")
    window: int = Field(5, description="Janela (ímpar)")
    poly: int = Field(2, description="Grau do polinômio (>=0)")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/process")
def process(d: DataIn):
    """
    Suaviza a série com Savitzky-Golay.
    Regras:
      - window precisa ser ímpar e >= 3
      - window não pode ser maior que o tamanho da série
      - poly < window
    """
    arr = np.array(d.data, dtype=float)
    n = len(arr)
    if n == 0:
        return {"error": "lista de dados vazia"}

    # Corrigir window para ímpar e válido
    w = int(d.window) if d.window else 5
    if w < 3: 
        w = 3
    if w % 2 == 0:
        w += 1
    if w > n:
        w = n if n % 2 == 1 else n - 1
    if w < 3:
        return {"error": "tamanho da série muito pequeno para suavização"}

    # Corrigir poly
    p = int(d.poly) if d.poly is not None else 2
    if p < 0:
        p = 0
    if p >= w:
        p = max(0, w - 1)

    smoothed = savgol_filter(arr, w, p).tolist()
    return {"smoothed": smoothed, "window": w, "poly": p}

@app.get("/")
def root():
    return {"service": "railsight-proc-python", "ok": True}
