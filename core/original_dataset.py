import pandas as pd


class OriginalDataset:
    def __init__(self, dataset_path) -> None:
        self.dataset = pd.read_csv(dataset_path)
        print(self.dataset.columns)

        self.reduced_dataset = self.dataset["cigever"]
        print(self.reduced_dataset.head())


if __name__ == "__main__":
    dataset = OriginalDataset("./data/original_dataset.csv")
