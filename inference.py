from fastapi import FastAPI

app = FastAPI()

@app.post("/reset")
def reset():
    return {"status": "ok"}

@app.get("/")
def home():
    return {"status": "running"}
