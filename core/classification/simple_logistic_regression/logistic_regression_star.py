import os
import pandas as pd
import logging

from sklearn.linear_model import LogisticRegression
from constants.columns import DatasetColumns
from constants.seed import RANDOM_SEED
from constants.columns import StellarClassOneHotEncoded

logger = logging.getLogger(__name__)

TRAIN_SPLIT_RATIO = 0.7


class LogisticRegressionModelStar:
    def __init__(self, data_folder):
        self.dataset = pd.read_csv(
            os.path.join(data_folder, "preprocessed/preprocessed_dataset.csv")
        )

        # Train and evaluate the model
        self.train_test_split()
        self.train_classifier()

    def train_test_split(self):
        """
        Split the dataset into a train and test set
        """

        self.train_set = self.dataset.sample(
            frac=TRAIN_SPLIT_RATIO, random_state=RANDOM_SEED
        )
        self.test_set = self.dataset.drop(self.train_set.index)

    def train_classifier(self):
        """
        Train the model
        """

        covariates = self.train_set.drop(
            columns=[
                StellarClassOneHotEncoded.STAR.value,
                StellarClassOneHotEncoded.GALAXY.value,
                StellarClassOneHotEncoded.QSO.value,
            ]
        )
        outcomes = self.train_set[StellarClassOneHotEncoded.STAR.value]

        self.classifier = LogisticRegression().fit(covariates, outcomes)

        # Log the accuracy on the train set (in-sample error)
        self.train_accuracy = ((self.classifier.predict(covariates) == outcomes)).mean()
        logger.info(
            f"Train accuracy for logistic regression model: {self.train_accuracy}"
        )

    def compute_test_accuracy(self):
        """
        Compute the accuracy on the test set
        """
        covariates = self.test_set.drop(
            columns=[
                StellarClassOneHotEncoded.STAR.value,
                StellarClassOneHotEncoded.GALAXY.value,
                StellarClassOneHotEncoded.QSO.value,
            ]
        )

        outcomes = self.test_set[StellarClassOneHotEncoded.STAR.value]

        test_accuracy = ((self.classifier.predict(covariates) == outcomes)).mean()

        logger.info(f"Test accuracy for logistic regression model: {test_accuracy}")
