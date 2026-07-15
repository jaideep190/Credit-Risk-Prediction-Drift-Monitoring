"""
Request/response schemas for the credit risk API.

ApplicantFeatures mirrors the raw dataset columns exactly (before any
preprocessing) - the API applies the same imputation/capping the model was
trained with, so the caller should send raw, unprocessed values, including
nulls for missing income/dependents if genuinely unknown.
"""

from typing import Optional

from pydantic import BaseModel, Field


class ApplicantFeatures(BaseModel):
    RevolvingUtilizationOfUnsecuredLines: float = Field(
        ..., ge=0, description="Total balance on credit cards/lines divided by credit limits"
    )
    age: int = Field(..., gt=0, le=110, description="Applicant age in years")
    NumberOfTime30_59DaysPastDueNotWorse: int = Field(
        ..., ge=0, alias="NumberOfTime30-59DaysPastDueNotWorse",
        description="Number of times 30-59 days past due in the last 2 years"
    )
    DebtRatio: float = Field(..., ge=0, description="Monthly debt payments / monthly income")
    MonthlyIncome: Optional[float] = Field(None, ge=0, description="Monthly income; null if unknown")
    NumberOfOpenCreditLinesAndLoans: int = Field(..., ge=0)
    NumberOfTimes90DaysLate: int = Field(..., ge=0)
    NumberRealEstateLoansOrLines: int = Field(..., ge=0)
    NumberOfTime60_89DaysPastDueNotWorse: int = Field(
        ..., ge=0, alias="NumberOfTime60-89DaysPastDueNotWorse",
        description="Number of times 60-89 days past due in the last 2 years"
    )
    NumberOfDependents: Optional[float] = Field(None, ge=0, description="Number of dependents; null if unknown")

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "RevolvingUtilizationOfUnsecuredLines": 0.45,
                "age": 42,
                "NumberOfTime30-59DaysPastDueNotWorse": 0,
                "DebtRatio": 0.35,
                "MonthlyIncome": 5500,
                "NumberOfOpenCreditLinesAndLoans": 6,
                "NumberOfTimes90DaysLate": 0,
                "NumberRealEstateLoansOrLines": 1,
                "NumberOfTime60-89DaysPastDueNotWorse": 0,
                "NumberOfDependents": 2,
            }
        }


class PredictionResponse(BaseModel):
    default_probability: float
    is_high_risk: bool
    decision_threshold: float
    model_name: str


class ExplanationResponse(BaseModel):
    default_probability: float
    is_high_risk: bool
    decision_threshold: float
    model_name: str
    base_value: float
    shap_contributions: dict