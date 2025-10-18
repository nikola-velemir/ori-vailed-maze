import glob
import os.path
import time

import pandas as pd

from src.generator.runner import run_training
from src.generator.utils import display_summary

if __name__ == "__main__":
    results = []
    print("Running test")
    for episode, file in enumerate(glob.glob("dataset/test/*.json"), start=1):
        print(f"{episode}. Episode")
        res = run_training(file, headless=True)
        results.append(res)

    print(results)
    df = pd.DataFrame(results)
    file_name = f'test_data_{time.time()}.csv'
    output = os.path.join("results", 'test', file_name)
    df.to_csv(output, index=False)

    display_summary(df=df)
