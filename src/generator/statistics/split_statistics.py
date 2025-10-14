import pandas as pd

from src.generator.statistics.statistics_utils import run_statistics


# Load the dataset
if __name__ == "__main__":


    df = pd.read_csv('results/test/test_data_1760371187.9622679.csv')
    run_statistics(df)
