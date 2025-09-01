"""
This script is used to standardize the data from dataset file.
"""

"""

Importing libraries

"""
import numpy as np
import pandas as pd
import logging 
# Configure logging for debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Standardization:
    def __init__(self):

        self.X = None
        self.y = None
        
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
        Xs,ys:               Data frame Standardize of X and y 
        Xprom, yprom :       Mean of X and y 
        Xstd, ystd:          Std desviation of X and y 
        Raises
        ValueError: If the arguments are not valid.
        -------
        '''
        """Improved input validation"""
        logger.debug("Validating inputs...")

        self.X = X.copy()
        self.y = y.copy()

        # Validate X and y
        if self.X is None or self.y is None:
            raise TypeError('X,y can\'t be None')
        
        # Validate types
        valid_types = (pd.DataFrame, np.ndarray)

        if not isinstance(self.X, valid_types):
            raise TypeError("X debe ser pandas.DataFrame o numpy.ndarray")

        if not isinstance(self.y, valid_types):
            raise TypeError("y debe ser pandas.DataFrame o numpy.ndarray")

        self.X = self.X.values if isinstance(self.X, pd.DataFrame) else self.X
        self.y = self.y.values if isinstance(self.y, pd.DataFrame) else self.y

        
        Xprom = np.mean(self.X, axis=0)
        Xstd = np.std(self.X, axis=0)
        Xs = (self.X-Xprom)/Xstd

        # Calculate the mean and standard deviation of the target.
        yprom = np.mean(self.y, axis=0)
        ystd = np.std(self.y, axis=0)
        ys = (self.y-yprom)/ystd

        Xs = pd.DataFrame(Xs)
        ys = pd.DataFrame(ys)

        return Xs,ys,yprom,ystd,Xprom,Xstd





    