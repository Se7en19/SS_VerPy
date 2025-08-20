"""
This script is used to split the data from a dataset file.
"""

""" Importing libraries """
import pandas as pd
import numpy as np

""" MAIN """

class SplitData:
    def __init__(self, X, y, test_size, random_state=None):
        """
        Verify if the DataFrame is a DataFrame. lol 
        
        """
        if not (isinstance(X, pd.DataFrame) ) and not( isinstance(y, pd.DataFrame)):
            raise TypeError('X and y must be a DataFrame')
        self.X = X.copy()
        self.y = y.copy()
        self.test_size = test_size
        self.random_state = random_state

    def split_data(self):
        """
        This function splits the data into training and testing sets.

        Returns
        X_train, X_test, y_train, y_test: DataFrames with the training and testing sets.
        
        """
        # Generate random indices for splitting the data
        if self.random_state is not None:
            np.random.seed(self.random_state)
        indices = np.random.permutation(len(self.X))
        split = int(len(self.X) * self.test_size)

        # Split the data into training and testing sets
        X_train = self.X.iloc[indices[:split]]
        X_test = self.X.iloc[indices[split:]]
        y_train = self.y.iloc[indices[:split]]
        y_test = self.y.iloc[indices[split:]]

        return X_train, X_test, y_train, y_test
