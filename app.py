from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np
from scipy.signal import savgol_filter

app = FastAPI(title="RailSight Processor", version="1.0.0")

# CORS aberto (ajuste depois se quiser restringir)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

class SeriesIn(BaseModel):
    curvature: Optional[List[float]] = None
    crosslevel_mm: Optional[List[float]] = None
    gauge_mm: Optional[List[float]] = None

class ProcessIn(BaseModel):
    km_ini: float = Field(..., description="km de referência real (ex.: 333800 m -> 333.800 km)")
    window_m: int = 300
    step_m: int = 1
    series: SeriesIn
    # parâmetros de suavização
    smooth_window: Optional[int] = Field(None, description="tamanho da janela Savitzky-Golay (ímpar)")
    polyorder: int = 2

class SeriesOut(BaseModel):
    curvature: Optional[List[float]] = None
    crosslevel_mm: Optional[List[float]] = None
    gauge_mm: Optional[List[float]] = None

class StatsOut(BaseModel):
    min: float
    max: float
    mean: float
    std: float

class ProcessOut(BaseModel):
    km_ini: float
    window_m: int
    step_m: int
    x_rel: List[float]
    series: SeriesOut
    stats: Dict[str, StatsOut]

@app.get("/health")
def health():
    return {"status": "ok"}

def _smooth(arr: Optional[List[float]], smooth_window: Optional[int], polyorder: int) -> Optional[List[float]]:
    if arr is None:
        return None
    a = np.array(arr, dtype=float)
    # janela default: ~1/15 do sinal, ímpar, mínimo 5
    if smooth_window is None:
        w = max(5, (len(a) // 15) | 1)  # força ímpar
    else:
        w = smooth_window if smooth_window % 2 == 1 else smooth_window + 1
        w = max(5, min(w, len(a) - (1 - len(a) % 2)))  # não exceder tamanho
    if len(a) < w:
        return a.tolist()
    try:
        return savgol_filter(a, window_length=w, polyorder=min(polyorder, w - 1)).tolist()
    except Exception:
        return a.tolist()

def _stats(arr: Optional[List[float]]) -> StatsOut:
    if arr is None or len(arr) == 0:
        return StatsOut(min=0, max=0, mean=0, std=0)
    a = np.array(arr, dtype=float)
    return StatsOut(min=float(np.min(a)), max=float(np.max(a)),
                    mean=float(np.mean(a)), std=float(np.std(a, ddof=1) if len(a) > 1 else 0.0))

@app.post("/process", response_model=ProcessOut)
def process(payload: ProcessIn):
    # eixo X relativo em metros (0..window-1)
    x_rel = list(range(0, payload.window_m, payload.step_m))

    # suavização
    curv = _smooth(payload.series.curvature, payload.smooth_window, payload.polyorder)
    cross = _smooth(payload.series.crosslevel_mm, payload.smooth_window, payload.polyorder)
    gauge = _smooth(payload.series.gauge_mm, payload.smooth_window, payload.polyorder)

    # truncar/igualar tamanho com base em x_rel
    n = len(x_rel)
    def fit_len(a):
        if a is None: return None
        m = min(n, len(a))
        return a[:m] + ([a[-1]] * (n - m)) if len(a) < n else a[:n]

    curv = fit_len(curv)
    cross = fit_len(cross)
    gauge = fit_len(gauge)

    stats = {}
    if curv is not None:  stats["curvature"]   = _stats(curv)
    if cross is not None: stats["crosslevel"]  = _stats(cross)
    if gauge is not None: stats["gauge"]       = _stats(gauge)

    return ProcessOut(
        km_ini=payload.km_ini,
        window_m=payload.window_m,
        step_m=payload.step_m,
        x_rel=x_rel,
        series=SeriesOut(curvature=curv, crosslevel_mm=cross, gauge_mm=gauge),
        stats=stats
    )

# Execução local (opcional): uvicorn app:app --reload
