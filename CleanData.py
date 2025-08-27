"""
This script is used to clean the data from dataset file.

"""

""" Importing libraries """
import pandas as pd
import numpy as np
""" MAIN """

class CleanData:
    def __init__(self):
       self.df = None
   
    def clean_nans(self,df, axis=0, method='drop', fill_value=None):
        ''' 
        This function cleans the NaNs from the DataFrame.

        Parameters
        ----------
        axis : int, optional
            Axis to clean the NaNs, by default 0
        method : str, optional
            Method to clean the NaNs, by default 'drop'
        fill_value : int, optional
            Value to fill the NaNs, by default None

        Returns
        pd.DataFrame: Clean DataFrame.

        Raises
        ValueError: If the arguments are not valid.
        -------
        '''
        self.df = df.copy()
        # Validate X and y
        if self.df is None:
            raise TypeError('The input cant be None')
        
        # Validate types
        valid_types = (pd.DataFrame)

        if not isinstance(self.df, valid_types):
            raise TypeError("X debe ser pandas.DataFrame")

        # Validate axis and method
        if axis not in (0, 1):
            raise ValueError('axis must be 0 (rows) or 1 (columns)')
        if method not in ['drop', 'fill']:
            raise ValueError('Method must be either "drop" or "fill"')

        # Apply cleaning
        if method == 'drop':
            cleaned_df = df.dropna(axis=axis)
        else:
            if fill_value is None:
                raise ValueError('fill_value must be specified')
            cleaned_df = df.fillna(fill_value)

        self.df = cleaned_df
        return cleaned_df
        