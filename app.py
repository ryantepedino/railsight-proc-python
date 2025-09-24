from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from scipy.signal import savgol_filter

app = FastAPI(title="RailSight Proc", version="1.0.0")

class SeriesIn(BaseModel):
    data: list[float]
    window: int | None = 11     # deve ser ímpar
    poly: int | None = 2        # < window

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/process")
def process(inp: SeriesIn):
    arr = np.array(inp.data, dtype=float)
    if arr.size < 5:
        return {"smoothed": inp.data, "window": None, "poly": None, "note": "poucos pontos"}

    # janela precisa ser ímpar e menor que len(arr)
    w = inp.window if inp.window else 11
    if w % 2 == 0:
        w += 1
    w = max(5, min(len(arr) - (1 - len(arr) % 2), w))  # garante ímpar e >=5
    if w % 2 == 0:
        w -= 1

    # poly < w
    poly = inp.poly if inp.poly is not None else 2
    poly = max(1, min(poly, w - 1))

    sm = savgol_filter(arr, window_length=w, polyorder=poly, mode="interp")
    return {"smoothed": sm.tolist(), "window": w, "poly": poly}
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
