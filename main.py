# main.py
from fastapi import FastAPI
import pickle
from pydantic import BaseModel

# بارگذاری مدل
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

# تعریف کلاس برای داده‌های ورودی
class IrisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# ایجاد endpoint برای پیش‌بینی
@app.post("/predict")
def predict(iris: IrisData):
    data = [[iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width]]
    prediction = model.predict(data)[0]
    return {"prediction": int(prediction)}
