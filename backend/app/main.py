from fastapi import FastAPI

app = FastAPI(title="VidSense AI Backend")

@app.get("/health")
def health():
    return {"status": "ok"}
