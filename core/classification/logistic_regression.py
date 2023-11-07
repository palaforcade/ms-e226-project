import os
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class LogisticRegressionModel:
    def __init__(self, data_folder):
        self.dataset = pd.read_csv(
            os.path.join(data_folder, "preprocessed/preprocessed_dataset.csv")
        )

        # Train and evaluate the model
        self.train_test_split()
        self.train_model()

    def train_test_split(self):
        """
        Split the dataset into a train and test set
        """
        pass

    def train_model(self):
        """
        Train the model
        """

        pass

    def compute_test_accuracy(self):
        """
        Compute the accuracy on the test set
        """

        logger.info(f"Test accuracy for logistic regression model: {'THE_RESULT'}")

        pass
