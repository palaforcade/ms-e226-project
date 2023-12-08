import os
import pandas as pd
import numpy as np
import logging
import statsmodels.api as sm

from constants.columns import DatasetColumns
from constants.seed import RANDOM_SEED

TRAIN_SPLIT_RATIO = 0.7

logger = logging.getLogger(__name__)


class OLSBaselineModel:
    def __init__(self, data_folder) -> None:
        self.dataset = pd.read_csv(
            os.path.join(data_folder, "preprocessed/preprocessed_dataset.csv")
        )

        # Train the model
        self.train_test_split()
        self.train_model()

    def train_test_split(self):
        """
        Split the dataset into a train and test set
        """
        self.train_set = self.dataset.sample(
            frac=TRAIN_SPLIT_RATIO, random_state=RANDOM_SEED
        )
        self.test_set = self.dataset.drop(self.train_set.index)

    def train_model(self):
        """
        Train the model using statsmodels
        """
        covariates = self.train_set.drop(columns=[DatasetColumns.REDSHIFT.value])
        outcomes = self.train_set[DatasetColumns.REDSHIFT.value]

        # Adding a constant for the intercept term
        covariates = sm.add_constant(covariates)

        self.model = sm.OLS(outcomes, np.asarray(covariates, dtype=float)).fit()

        # Log the MSE on the train set (in-sample error)
        predictions = self.model.predict(covariates)
        self.train_mse = ((predictions - outcomes) ** 2).mean()
        logger.info(f"Train MSE for OLS baseline model: {self.train_mse}")

        print("SUMMARY FOR THE BASELINE MODEL")
        print(self.model.summary2())

    def compute_test_mse(self):
        """
        Compute the MSE on the test set
        """
        covariates = self.test_set.drop(columns=[DatasetColumns.REDSHIFT.value])
        outcomes = self.test_set[DatasetColumns.REDSHIFT.value]

        # Adding a constant for the intercept term
        covariates = sm.add_constant(covariates)

        predictions = self.model.predict(covariates)
        self.test_mse = ((predictions - outcomes) ** 2).mean()

        logger.info(f"Test MSE for OLS baseline model: {self.test_mse}")

    def compute_coefficient_p_values(self):
        """
        Compute the p-values for all the linear regression coefficients
        using statsmodels
        """
        # The p-values are directly available in the summary
        model_summary = self.model.summary2()
        self.p_values = self.model.summary2().tables[1]["P>|t|"]
        return model_summary
    
    def print_significant_coefficients(self):
        if self.model is None:
            raise ValueError("Fit the model first using the fit method.")

        # Filter coefficients based on p-value
        significant_coeffs = self.model.params[self.model.pvalues < self.threshold]

        # Print significant coefficients
        print("Significant Coefficients:")
        for coeff_name, coeff_value in significant_coeffs.items():
            print(f"{coeff_name}: {coeff_value}")
