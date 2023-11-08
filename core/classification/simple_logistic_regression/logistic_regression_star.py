import os
import pandas as pd
import logging

from sklearn.linear_model import LogisticRegression
from constants.columns import DatasetColumns
from constants.seed import RANDOM_SEED

logger = logging.getLogger(__name__)

TRAIN_SPLIT_RATIO = 0.7

class LogisticRegressionModel:
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

        covariates = self.train_set.drop(columns=[DatasetColumns.REDSHIFT.value])
        outcomes = (self.train_set[DatasetColumns.CLASS.value] == "STAR").astype(int)

        self.classifier = LogisticRegression().fit(covariates, outcomes)

    def predict_class(self, new_observation):
        """
        Predict the class of a new observation
        """
        new_covariates = new_observation.drop(columns=[DatasetColumns.REDSHIFT.value])
        predicted_prob = self.classifier.predict_proba(new_covariates)[:, 1]  # Probability of being a star
        return predicted_prob
    
    def compute_test_accuracy(self):
        """
        Compute the accuracy on the test set
        """
        covariates = self.test_set.drop(columns=[DatasetColumns.REDSHIFT.value])
        outcomes = (self.train_set[DatasetColumns.CLASS.value] == "Star").astype(int)

        self.test_mse = ((self.classifier.predict(covariates) - outcomes) ** 2).mean()

        logger.info(f"Test accuracy for logistic regression model: {self.test_mse}")
