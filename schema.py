from pydantic import BaseModel, Field, StrictFloat, StrictInt, StrictStr
from typing import List


class CustomerInput_Schema(BaseModel):
    LIMIT_BAL: float
    PAY_SEPT: int
    PAY_AUG: int
    PAY_JUL: int
    PAY_JUN: int
    PAY_MAY: int
    PAY_APR: int

    BILL_JUL: float
    BILL_AUG: float
    BILL_MAY: float
    BILL_JUN: float
    BILL_APR: float
    BILL_SEPT: float

    PAY_AMT_AUG: float
    PAY_AMT_SEPT: float
    PAY_AMT_MAY: float
    PAY_AMT_JUL: float
    PAY_AMT_JUN: float
    PAY_AMT_APR: float


class Prediction_OutputSchema(BaseModel):
    prediction: str
    risk_probability: float
    risk_level: str
    key_reasons: List[str]
    fairness_note: str
