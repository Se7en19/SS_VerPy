"""
This script is used to standardize the data from dataset file.
"""

"""

Importing libraries

"""
import numpy as np
import pandas as pd

class Standardization:
    def __init__(self, X,y):
        '''
        Verify if X and y are DataFrames. lol
        
        '''
        if not isinstance(X, pd.DataFrame) and not isinstance(y, pd.DataFrame):
            raise TypeError('X and y must be a DataFrame')
        self.X=X.copy()
        self.y=y.copy()

    def standardize(self,X,y):
        '''
        This function standardizes the data.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame with the features.
        y : pd.DataFrame
            DataFrame with the target.

        Returns
        X,y: DataFrame with the standardized data.

        Raises
        ValueError: If the arguments are not valid.
        -------
        '''
        
        # Calculate the mean and standard deviation of each feature.
        Xval = X.values
        Xprom = np.mean(Xval, axis=0)
        Xstd = np.std(Xval, axis=0)
        Xs = (X-Xprom)/Xstd

        # Calculate the mean and standard deviation of the target.
        yval = y.values
        yprom = np.mean(yval, axis=0)
        ystd = np.std(yval, axis=0)
        ys = (y-yprom)/ystd

        return Xs,ys





    