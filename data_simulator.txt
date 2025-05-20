
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from threading import Thread
import time
import numpy as np

app = FastAPI()

# ------------------ Simulation State ------------------
BASE_INDEX = 23000
INTERVAL = 2  # seconds
t = 0
latest_price = BASE_INDEX
status_flag = "T"

# ------------------ Price Generator ------------------
def price_simulator():
    global t, latest_price, status_flag
    while True:
        noise = np.random.normal(0, 0.02)
        wave = np.sin(t / 30)
        trend = 0.0005 * t
        price = (wave + noise + trend) * 100 + BASE_INDEX
        latest_price = round(price, 2)
        status_flag = "T"
        t += 1
        time.sleep(INTERVAL)

# ------------------ API Endpoint ------------------
@app.get("/latest")
def get_latest_price():
    return JSONResponse({
        "timestamp": time.time(),
        "price": latest_price,
        "status": status_flag
    })

# ------------------ Background Simulation ------------------
def start_background_simulator():
    thread = Thread(target=price_simulator, daemon=True)
    thread.start()

# ------------------ Startup Event ------------------
@app.on_event("startup")
def on_startup():
    start_background_simulator()
