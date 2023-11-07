import os
import logging
import pandas as pd
from sklearn.linear_model import Lasso

from constants.columns import DatasetColumns
from constants.seed import RANDOM_SEED


TRAIN_SPLIT_RATIO = 0.7

logger = logging.getLogger(__name__)


class LassoWithSquaresModel:
    LASSO_ALPHA = 0.3

    def __init__(self, data_folder) -> None:
        self.dataset = pd.read_csv(
            os.path.join(data_folder, "preprocessed/preprocessed_dataset.csv")
        )

        # Train the model
        self.add_engineered_features()
        self.train_test_split()
        self.train_model()

    def add_engineered_features(self):
        # Add squared terms for MAGNITUDES AND PSF_MAGNITUDES
        cols_to_square = [
            DatasetColumns.MAGNITUDE_FIT_G,
            DatasetColumns.MAGNITUDE_FIT_R,
            DatasetColumns.MAGNITUDE_FIT_I,
            DatasetColumns.MAGNITUDE_FIT_Z,
            DatasetColumns.MAGNITUDE_FIT_U,
            DatasetColumns.PSF_MAGNITUDE_G,
            DatasetColumns.PSF_MAGNITUDE_R,
            DatasetColumns.PSF_MAGNITUDE_I,
            DatasetColumns.PSF_MAGNITUDE_Z,
            DatasetColumns.PSF_MAGNITUDE_U,
        ]

        for col in cols_to_square:
            self.dataset[f"{col.value}_squared"] = self.dataset[col.value] ** 2

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

        self.model = Lasso(alpha=self.LASSO_ALPHA).fit(covariates, outcomes)

    def compute_test_mse(self):
        """
        Compute the MSE on the test set
        """

        covariates = self.test_set.drop(columns=[DatasetColumns.REDSHIFT.value])
        outcomes = self.test_set[DatasetColumns.REDSHIFT.value]

        self.test_mse = ((self.model.predict(covariates) - outcomes) ** 2).mean()

        logger.info(f"Test MSE for Lasso with squares model: {self.test_mse}")
