import os
import pandas as pd
import logging
from torch import nn, optim, from_numpy, no_grad
from sklearn.preprocessing import StandardScaler

from constants.seed import RANDOM_SEED
from constants.columns import DatasetColumns
from regression.one_layer_perceptron.nn_model import NNModel

TRAIN_SPLIT_RATIO = 0.7
HIDDEN_LAYER_SIZE = 15
LEARNING_RATE = 0.01
EPOCHS_COUNT = 10

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
        self.standardize_covariates()
        self.convert_to_tensor()
        self.train_model()

    def train_test_split(self):
        """
        Split the dataset into a train and test set
        """

        self.train_set = self.dataset.sample(
            frac=TRAIN_SPLIT_RATIO, random_state=RANDOM_SEED
        )
        self.test_set = self.dataset.drop(self.train_set.index)

        ## Separate the covariates and the outcomes

        self.train_covariates = self.train_set.drop(
            columns=[DatasetColumns.REDSHIFT.value]
        )
        self.train_outcomes = self.train_set[DatasetColumns.REDSHIFT.value]

        self.test_covariates = self.test_set.drop(
            columns=[DatasetColumns.REDSHIFT.value]
        )
        self.test_outcomes = self.test_set[DatasetColumns.REDSHIFT.value]

    def standardize_covariates(self):
        """
        Standardize the covariates
        """

        self.scaler = StandardScaler()

        self.scaler.fit(self.train_covariates)

        self.train_covariates = self.scaler.transform(self.train_covariates)
        self.test_covariates = self.scaler.transform(self.test_covariates)

    def convert_to_tensor(self):
        """
        Convert the train and test sets to pytorch tensors
        """

        self.train_covariates = from_numpy(self.train_covariates).float()

        self.train_outcomes = from_numpy(self.train_outcomes.values).float()

        self.test_covariates = from_numpy(self.test_covariates).float()

        self.test_outcomes = from_numpy(self.test_outcomes.values).float()

    def train_model(self):
        """
        Define and train the perceptron using pytorch
        """
        self.model = NNModel(self.train_covariates.shape[1], HIDDEN_LAYER_SIZE)

        # Define the loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.SGD(self.model.parameters(), lr=LEARNING_RATE)

        # Training loop
        for epoch in range(EPOCHS_COUNT):
            # Forward pass
            y_pred = self.model(self.train_covariates)

            # Compute and print loss
            loss = criterion(y_pred, self.train_outcomes)
            logger.info(f"Epoch {epoch + 1} | Loss: {loss.item():.4f}")

            # Zero gradients, perform a backward pass, and update the weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def compute_test_mse(self):
        """
        Compute the MSE on the test set
        """

        with no_grad():
            y_pred = self.model(self.test_covariates)
            mse = nn.MSELoss()
            test_mse = mse(y_pred, self.test_outcomes)
            logger.info(f"Test MSE for one-layer perceptron: {test_mse.item():.4f}")
