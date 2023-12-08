import os
import pandas as pd
import logging
from torch import nn, optim, from_numpy, no_grad
from sklearn.preprocessing import StandardScaler

from constants.seed import RANDOM_SEED
from constants.columns import DatasetColumns

TRAIN_SPLIT_RATIO = 0.7
LEARNING_RATE = 0.01
EPOCHS_COUNT = 1000

logger = logging.getLogger(__name__)


class Perceptron:
    """
    A model to predict the redshift of a stellar object using a neural network.
    """

    def __init__(self, data_folder, model_structure):
        self.dataset = pd.read_csv(
            os.path.join(data_folder, "preprocessed/preprocessed_dataset.csv")
        )
        self.data_folder = data_folder

        self.model_structure = model_structure

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
        self.model = self.model_structure(self.train_covariates.shape[1])

        # Define the loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

        # Training loop
        for epoch in range(EPOCHS_COUNT):
            # Forward pass
            y_pred = self.model(self.train_covariates)

            # Compute and print loss
            loss = criterion(y_pred.squeeze(), self.train_outcomes)
            logger.debug(f"Epoch {epoch + 1} | Loss: {loss.item():.4f}")

            # Zero gradients, perform a backward pass, and update the weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Log the MSE on the train set (in-sample error)
        train_mse = nn.MSELoss()
        train_mse = train_mse(y_pred.squeeze(), self.train_outcomes)
        logger.info(
            f"Train MSE for {self.model_structure.__name__} NN: {train_mse.item():.4f}"
        )

    def compute_test_mse(self):
        """
        Compute the MSE on the test set
        """

        with no_grad():
            y_pred = self.model(self.test_covariates)
            mse = nn.MSELoss()
            test_mse = mse(y_pred.squeeze(), self.test_outcomes)
            logger.info(
                f"Test MSE for {self.model_structure.__name__} NN: {test_mse.item():.4f}"
            )

    def compute_holdout_mse(self):
        """
        Compute the MSE on the holdout set
        """

        holdout_dataset = pd.read_csv(
            os.path.join(
                self.data_folder, "preprocessed/preprocessed_holdout_dataset.csv"
            )
        )

        holdout_covariates = holdout_dataset.drop(
            columns=[DatasetColumns.REDSHIFT.value]
        )

        holdout_covariates = self.scaler.transform(holdout_covariates)

        holdout_covariates = from_numpy(holdout_covariates).float()

        holdout_outcomes = holdout_dataset[DatasetColumns.REDSHIFT.value]

        holdout_outcomes = from_numpy(holdout_outcomes.values).float()

        with no_grad():
            y_pred = self.model(holdout_covariates)
            mse = nn.MSELoss()
            holdout_mse = mse(y_pred.squeeze(), holdout_outcomes)
            logger.info(
                f"Holdout MSE for {self.model_structure.__name__} NN: {holdout_mse.item():.4f}"
            )
