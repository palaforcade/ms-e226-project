import os
import pandas as pd
import numpy as np
import logging
import statsmodels.api as sm

from constants.columns import DatasetColumns
from constants.seed import RANDOM_SEED

TRAIN_SPLIT_RATIO = 0.7

logger = logging.getLogger(__name__)


class OLSSubsetModel:
    def __init__(self, data_folder) -> None:
        full_dataset = pd.read_csv(
            os.path.join(data_folder, "preprocessed/preprocessed_dataset.csv")
        )

        # Keep only the 25 first columns in the dataset and the last
        self.dataset = full_dataset.iloc[:, :25]

        # Readd the redshift column
        self.dataset[DatasetColumns.REDSHIFT.value] = full_dataset[
            DatasetColumns.REDSHIFT.value
        ]

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
        print("SUMMARY FOR THE SUBSET MODEL")
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
        self.p_values = self.model.summary2().tables[1]["P>|t|"]
        return self.p_values
