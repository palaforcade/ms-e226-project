import matplotlib.pyplot as plt
import numpy as np
from regression.lasso_with_squares import LassoWithSquaresModel

LAMBDA_RANGE = np.linspace(0.01, 1.0, 100)


class LassoLambdaSearch:
    def __init__(self, data_folder, lambda_range):
        self.data_folder = data_folder
        self.lambda_range = lambda_range
        self.mses = []

    def train_and_plot_mses(self):
        for alpha in self.lambda_range:
            model = LassoWithSquaresModel(data_folder=self.data_folder)
            model.LASSO_ALPHA = alpha  # Set the alpha value for Lasso

            # Train the model with the specified alpha
            model.train_model()
            model.compute_test_mse()

            self.mses.append(model.test_mse)

        self.plot_mses()

    def plot_mses(self):
        plt.plot(self.lambda_range, self.mses)
        plt.xlabel("Lambda (Alpha) Values")
        plt.ylabel("Test MSE")
        plt.title("MSE vs. Lambda (Alpha)")
        plt.show()


# if __name__ == "__main__":
#     lambda_range = LAMBDA_RANGE  # Adjust the range as needed
#     data_folder = "your_data_folder_path_here"  # Provide the path to your data folder
#     lasso_lambda_search = LassoLambdaSearch(data_folder, lambda_range)
#     lasso_lambda_search.train_and_plot_mses()
