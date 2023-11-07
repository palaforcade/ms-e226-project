import os
import logging
import pandas as pd
from sklearn.linear_model import LinearRegression

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

    def compute_test_mse(self):
        """
        Compute the MSE on the test set
        """

        covariates = self.test_set.drop(columns=[DatasetColumns.REDSHIFT.value])
        outcomes = self.test_set[DatasetColumns.REDSHIFT.value]

        self.test_mse = ((self.model.predict(covariates) - outcomes) ** 2).mean()

        logger.info(f"Test MSE: {self.test_mse}")
