# src/main.py

from fastapi import FastAPI

# Create FastAPI app
app = FastAPI()

@app.get("/ping")
def ping():
    return {"message": "pong"}
