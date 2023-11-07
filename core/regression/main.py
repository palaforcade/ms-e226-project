import logging
from regression.ols_model import OLSBaselineModel

logger = logging.getLogger(__name__)


def run_regression_models(data_folder):
    """
    Run the regression models
    """
    logger.info("Running regression models")

    OLSBaselineModel(data_folder).compute_test_mse()

    logger.info("Regression models done")
