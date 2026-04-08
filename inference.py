from fastapi import FastAPI

app = FastAPI()

@app.post("/reset")
def reset():
    return {"message": "Environment reset successful"}

@app.get("/")
def home():
    return {"message": "API is working"}
