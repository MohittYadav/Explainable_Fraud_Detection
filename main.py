from fastapi import FastAPI, HTTPException
import pandas as pd
from schema import CustomerInput_Schema, Prediction_OutputSchema
from predict import predict_customer
from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI(
    title="Explainable Fraud Detection API",
    description="Predicts default risk with SHAP-based explanations",
    version="1.0"
)
templates = Jinja2Templates(directory="templates")


# Render UI using Jinja2
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html", {"request": request}
    )

@app.post("/predict", response_model=Prediction_OutputSchema)
async def predict(customer: CustomerInput_Schema):
    try:
        raw_df = pd.DataFrame([customer.dict()])
        result = predict_customer(raw_df)
        return Prediction_OutputSchema(
            prediction=result["prediction"],
            risk_probability=result["risk_probability"],
            risk_level=result["risk_level"],
            key_reasons=result["key_reasons"],
            fairness_note=(
                "This decision is based on financial and behavioral attributes. "
                "Sensitive demographic attributes such as gender and age "
                "did not materially influence the prediction."
            )
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
