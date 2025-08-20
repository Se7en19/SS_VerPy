"""
This script is used to clean the data from dataset file.

"""

""" Importing libraries """
import pandas as pd
""" MAIN """

class CleanData:
    def __init__(self, df):
        '''
        Verify if the DataFrame is a DataFrame. lol 
        
        '''
        if not isinstance(df, pd.DataFrame):
            raise TypeError('DataFrame expected')
        self.df=df.copy()
   
    def clean_nans(self, axis=0, method='drop', fill_value=None):
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
        pd.DataFrame: DataFrame limpio.

        Raises
        ValueError: If the arguments are not valid.
        -------
        '''
        if method not in ['drop', 'fill']:
            raise ValueError('Method must be either "drop" or "fill"')
        if method == 'drop':
           cleaned_df= self.df.dropna(axis=axis)
        else:
            if fill_value is None:
                raise ValueError('fill_value must be specified')
            cleaned_df =self.df.fillna(fill_value)
        return cleaned_df
        
    def get_cleaned_data(self):

        ''' 
        This function returns the cleaned DataFrame.
        
        Returns
        pd.DataFrame: DataFrame limpio.
        
        '''

        return self.df