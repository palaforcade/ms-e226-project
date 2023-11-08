import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler

from constants.seed import RANDOM_SEED
from constants.columns import DatasetColumns


class NeuralNetworkClassifier:
    def __init__(self, data_folder) -> None:
        self.dataset = pd.read_csv(
            os.path.join(data_folder, "preprocessed/preprocessed_dataset.csv")
        )

        # Train the classifier
        self.train_test_split()
        self.build_and_train_model()

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

    def build_and_train_model(self):
        """
        Build and train the neural network classifier
        """

        covariates = self.train_set.drop(columns=[DatasetColumns.REDSHIFT.value])
        outcomes = (self.train_set[DatasetColumns.CLASS.value] == "Star").astype(int)

        # Convert data to PyTorch tensors
        covariates = torch.Tensor(covariates.values)
        outcomes = torch.Tensor(outcomes.values).unsqueeze(1)

        # Define a simple feedforward neural network
        input_size = covariates.shape[1]
        model = nn.Sequential(
            nn.Linear(input_size, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
        )

        # Define the loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train the model
        for epoch in range(10):
            optimizer.zero_grad()
            outputs = model(covariates)
            loss = criterion(outputs, outcomes)
            loss.backward()
            optimizer.step()

        self.classifier = model

    def predict_class(self, new_observation):
        """
        Predict the class of a new observation
        """
        covariates = new_observation.drop(columns=[DatasetColumns.REDSHIFT.value])
        covariates = torch.Tensor(covariates.values)
        predicted_prob = self.classifier(covariates)
