from enum import Enum


class Task(Enum):
    PREPROCESSING = "preprocessing"
    EXPLORATION = "exploration"
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
