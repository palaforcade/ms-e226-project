import logging

from regression.ols_model import OLSBaselineModel
from regression.lasso_with_squares import LassoWithSquaresModel
from regression.lasso_lambda_selection import LassoLambdaSearch, LAMBDA_RANGE
from regression.neural_networks.perceptron import Perceptron
from regression.neural_networks.one_layer_model import OneLayerModel
from regression.neural_networks.multi_layer_model import MultiLayerModel

logger = logging.getLogger(__name__)


def run_regression_models(data_folder):
    """
    Run the regression models
    """
    logger.info("Running regression models")

    OLSBaselineModel(data_folder).compute_test_mse()

    LassoWithSquaresModel(data_folder).compute_test_mse()

    LassoLambdaSearch(data_folder, LAMBDA_RANGE)

    Perceptron(
        data_folder=data_folder, model_structure=OneLayerModel
    ).compute_test_mse()

    Perceptron(
        data_folder=data_folder, model_structure=MultiLayerModel
    ).compute_test_mse()

    logger.info("Regression models done")
