import logging

from regression.ols_model import OLSBaselineModel
from regression.lasso_with_squares import LassoWithSquaresModel
from regression.one_layer_perceptron.one_layer_perceptron import OneLayerPerceptron

logger = logging.getLogger(__name__)


def run_regression_models(data_folder):
    """
    Run the regression models
    """
    logger.info("Running regression models")

    OLSBaselineModel(data_folder).compute_test_mse()

    LassoWithSquaresModel(data_folder).compute_test_mse()

    OneLayerPerceptron(data_folder).compute_test_mse()

    logger.info("Regression models done")
