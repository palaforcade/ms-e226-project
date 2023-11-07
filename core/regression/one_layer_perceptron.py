import os
import pandas as pd
import logging

from constants.seed import RANDOM_SEED

TRAIN_SPLIT_RATIO = 0.7

logger = logging.getLogger(__name__)


class OneLayerPerceptron:
    """
    A model to predict the redshift of a stellar object using a neural network with one hidden layer
    """

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

        self.train_set = self.dataset.sample(
            frac=TRAIN_SPLIT_RATIO, random_state=RANDOM_SEED
        )
        self.test_set = self.dataset.drop(self.train_set.index)

    def train_model(self):
        """
        Define and train the perceptron using pytorch
        """
        pass

    def compute_test_mse(self):
        """
        Compute the MSE on the test set
        """

        logger.info(f"Test MSE for one-layer perceptron: {'THE_RESULT'}")
