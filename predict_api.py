from fastapi import FastAPI
import pickle

# Load model
with open('pipeline_v1.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

# Initialize FastAPI
app = FastAPI()

@app.get("/")
def home():
    return {"message": "Lead scoring model is running "}

@app.post("/predict")
def predict(client: dict):
    pred = pipeline.predict_proba([client])[0, 1]
    return {"probability": round(float(pred), 3)}
