import os
import pandas as pd
import numpy as np
import logging
import statsmodels.api as sm

from constants.columns import DatasetColumns
from constants.seed import RANDOM_SEED

TRAIN_SPLIT_RATIO = 0.7
SIGNIFICANCY_THRESHOLD = 0.001

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
        Train the model using statsmodels
        """
        covariates = self.train_set.drop(columns=[DatasetColumns.REDSHIFT.value])
        outcomes = self.train_set[DatasetColumns.REDSHIFT.value]

        # Adding a constant for the intercept term
        covariates = sm.add_constant(covariates)

        self.model = sm.OLS(outcomes, covariates.astype(float)).fit()

        # Log the MSE on the train set (in-sample error)
        predictions = self.model.predict(covariates)
        self.train_mse = ((predictions - outcomes) ** 2).mean()
        logger.info(f"Train MSE for OLS baseline model: {self.train_mse}")

        print("SUMMARY FOR THE BASELINE MODEL")
        print(self.model.summary2())

    def compute_test_mse(self):
        """
        Compute the MSE on the test set
        """
        covariates = self.test_set.drop(columns=[DatasetColumns.REDSHIFT.value])
        outcomes = self.test_set[DatasetColumns.REDSHIFT.value]

        # Adding a constant for the intercept term
        covariates = sm.add_constant(covariates)

        predictions = self.model.predict(covariates)
        self.test_mse = ((predictions - outcomes) ** 2).mean()

        logger.info(f"Test MSE for OLS baseline model: {self.test_mse}")

    def compute_coefficient_p_values(self):
        """
        Compute the p-values for all the linear regression coefficients
        using statsmodels
        """
        # The p-values are directly available in the summary
        model_summary = self.model.summary2()
        self.p_values = self.model.summary2().tables[1]["P>|t|"]
        return model_summary

    def significant_coefficients(self):
        if self.model is None:
            raise ValueError("Fit the model first using the fit method.")

        # Filter coefficients based on p-value
        significant_coeffs = self.model.params[
            self.model.pvalues < SIGNIFICANCY_THRESHOLD
        ]

        # Print significant coefficients
        logger.info("Computing significant coefficients for the OLS Baseline model")
        logger.info(
            f"Significant coefficients for the OLS baseline model: {list(significant_coeffs.index)}"
        )

    def compute_ci_using_bootstrap(self):
        """
        Compute the confidence intervals for the coefficients using bootstrap
        """

        # Number of bootstrap samples
        n_bootstrap = 1000

        # Sample size
        n = len(self.train_set)

        # Bootstrap samples
        bootstrap_samples = [
            self.train_set.sample(n=n, replace=True, random_state=seed)
            for seed in range(n_bootstrap)
        ]

        # Compute the coefficients for each bootstrap sample
        bootstrap_coefficients = [
            sm.OLS(
                sample[DatasetColumns.REDSHIFT.value],
                sm.add_constant(sample.drop(columns=[DatasetColumns.REDSHIFT.value])),
            )
            .fit()
            .params
            for sample in bootstrap_samples
        ]

        # Compute the confidence intervals
        self.confidence_intervals = pd.DataFrame(
            {
                "lower": [
                    np.quantile(
                        [bootstrap_coefficients[i][j] for i in range(n_bootstrap)],
                        q=0.025,
                    )
                    for j in range(len(bootstrap_coefficients[0]))
                ],
                "upper": [
                    np.quantile(
                        [bootstrap_coefficients[i][j] for i in range(n_bootstrap)],
                        q=0.975,
                    )
                    for j in range(len(bootstrap_coefficients[0]))
                ],
            },
            index=bootstrap_coefficients[0].index,
        )

        return self.confidence_intervals

    def benjamini_hochberg(self, alpha=0.001):
        """
        Apply the Benjamini-Hochberg procedure to control the FDR at level alpha.
        Returns a Boolean array where True indicates the hypotheses that are rejected.
        """
        m = len(self.p_values)  # Total number of hypotheses
        sorted_p_values = np.sort(self.p_values)
        sorted_index = np.argsort(self.p_values)
        rank = np.arange(1, m + 1)

        # Compute the threshold for each p-value
        threshold = (rank / m) * alpha

        # Find where p-value crosses the threshold
        reject = sorted_p_values <= threshold
        max_accept = np.max(np.where(reject)[0]) if np.any(reject) else 0

        # Apply the correction to the original p-values
        corrected_accept = np.zeros(m, dtype=bool)
        corrected_accept[sorted_index[: max_accept + 1]] = True

        # Log the rejected hypotheses along with their index
        logger.info(
            f"Significant coefficients according to the BH method: {list(self.p_values[corrected_accept].index)}"
        )

        return corrected_accept
