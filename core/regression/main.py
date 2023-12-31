import logging

from regression.ols_model_on_heldout import OLSModelOnHoldout
from regression.ols_model_subset_cov import OLSSubsetModel
from regression.ols_model import OLSBaselineModel
from regression.lasso_with_squares import LassoWithSquaresModel
from regression.neural_networks.perceptron import Perceptron
from regression.neural_networks.one_layer_model import OneLayerModel
from regression.neural_networks.multi_layer_model import MultiLayerModel

logger = logging.getLogger(__name__)


def run_regression_models(data_folder):
    """
    Run the regression models
    """
    logger.info("Running regression models")

    ols_model = OLSBaselineModel(data_folder)
    ols_model.compute_test_mse()
    ols_model.significant_coefficients()
    ols_model.compute_coefficient_p_values()
    print(ols_model.benjamini_hochberg())

    ols_subset_model = OLSSubsetModel(data_folder)
    ols_subset_model.compute_test_mse()
    ols_subset_model.significant_coefficients()

    ols_heldout_model = OLSModelOnHoldout(data_folder)
    ols_heldout_model.compute_test_mse()
    ols_heldout_model.significant_coefficients()
    ols_heldout_model.compute_ci_using_bootstrap()

    # LassoWithSquaresModel(data_folder).compute_test_mse()

    # LassoWithSquaresModel(data_folder).plot_mse_on_alpha_values()

    Perceptron(
        data_folder=data_folder, model_structure=OneLayerModel
    ).compute_test_mse()

    multi_layer_perceptron = Perceptron(
        data_folder=data_folder, model_structure=MultiLayerModel
    )

    multi_layer_perceptron.compute_test_mse()
    multi_layer_perceptron.compute_holdout_mse()

    logger.info("Regression models done")
