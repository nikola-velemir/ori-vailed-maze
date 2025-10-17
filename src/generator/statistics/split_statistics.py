import pandas as pd

from src.generator.statistics.statistics_utils import run_statistics


if __name__ == "__main__":


    df = pd.read_csv('results/test/test_data_1760707261.963145.csv')
    run_statistics(df)
