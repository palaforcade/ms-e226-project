import os
import numpy as np
import logging
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.stats import t
from torch import cov

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
        Train the model
        """

        covariates = self.train_set.drop(columns=[DatasetColumns.REDSHIFT.value])
        outcomes = self.train_set[DatasetColumns.REDSHIFT.value]

        self.model = LinearRegression().fit(covariates, outcomes)

        # Log the MSE on the train set (in-sample error)
        self.train_mse = ((self.model.predict(covariates) - outcomes) ** 2).mean()
        logger.info(f"Train MSE for OLS baseline model: {self.train_mse}")

    def compute_test_mse(self):
        """
        Compute the MSE on the test set
        """

        covariates = self.test_set.drop(columns=[DatasetColumns.REDSHIFT.value])
        outcomes = self.test_set[DatasetColumns.REDSHIFT.value]

        self.test_mse = ((self.model.predict(covariates) - outcomes) ** 2).mean()

        logger.info(f"Test MSE for OLS baseline model: {self.test_mse}")

    def compute_coefficient_p_values(self):
        """
        Compute the p-values for all the linear regression coefficients
        """

        covariates = (
            self.train_set.drop(columns=[DatasetColumns.REDSHIFT.value])
            .to_numpy()
            .astype(np.float64)
        )

        outcomes = self.train_set[DatasetColumns.REDSHIFT.value].to_numpy()

        n = len(self.train_set)
        p = covariates.shape[1]
        dof = n - p - 1

        # Compute the standard errors of the coefficients
        residuals = outcomes - self.model.predict(covariates)
        mse = (residuals**2).sum() / dof
        prod_matrix = np.dot(covariates.T, covariates)
        cov_matrix = mse * np.linalg.inv(prod_matrix)

        se = np.diag(cov_matrix) ** 0.5

        # Compute the t-statistics and p-values
        t_values = self.model.coef_ / se
        p_values = (1 - t.cdf(abs(t_values), dof)) * 2

        return p_values
