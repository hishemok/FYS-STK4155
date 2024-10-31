import pandas as pd
import numpy as np
def read_data():
    """
    Reads the data from the file and returns the data as a pandas dataframe.
    """
    data = pd.read_csv("Project2/data/data.csv")
    #convert to np array
    data = data.to_numpy()

    return data

if __name__ == "__main__":
    d = read_data()
    print(d)