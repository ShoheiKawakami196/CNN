from fastapi import FastAPI
from app.routers import prediction

app = FastAPI(
    title="Skin Type Classifier API",
    description="API for classifying skin types from images",
    version="0.1.0"
)

# ルーターの登録
app.include_router(prediction.router)

@app.get("/")
def read_root():
    return {"message": "Welcome to Skin Type Classifier API!"}
