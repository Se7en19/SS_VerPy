"""
This script is a regression model, using the PCR algorithm and my own perceptron algorithm.
by: Se7en19 
"""

""" Importing libraries """
import pandas as pd
import numpy as np
from pandas.core.base import NoNewAttributesMixin
import CleanData as cd 
import split_data as sd
import estandarizacion as st 
from sklearn.decomposition import PCA

""" MAIN """

class PCR:
    def __init__(self):
        """
        This function initializes the PCR class.
        
        """
        self.df = None
        self.X = None
        self.y = None
        self.test_size = None
        self.random_state = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.eta = None 
        self.numEpochs = None
        self.numComponents=None
        
    def PCR(self,X,y,eta,numEpochs,test_size,random_state,numComponents):
        """
        This function trains the PCR model.

        Parameters
        ----------
        X: DataFrame with the features.
        y: DataFrame with the target.
        eta: float with the learning rate.
        numEpochs: int with the number of epochs.
        test_size: float with the size of the testing set.
        random_state: int with the random state.

        Returns
        X_train, X_test, y_train, y_test: DataFrames with the training and testing sets.
        
        """
        if X or y is None:
            raise TypeError('The input (X,y) must be a pandas DataFrame or numpy array')
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        elif isinstance(y, np.ndarray):
            y = pd.DataFrame(y)
        elif not( isinstance(X, pd.DataFrame) and isinstance(y, pd.DataFrame)):
            raise TypeError('X and y must be a DataFrame')


        """ Variables """
        self.X = X.copy()
        self.y = y.copy()
        self.eta = eta
        self.numEpochs = numEpochs
        self.test_size = test_size
        self.random_state = random_state

        """ We used the CleanData class to clean the data """
        cd = cd.CleanData()
        self.X=cd.clean_nans(self.X)
        self.y=cd.clean_nans(self.y)

        ''' We used the split_data class to split the data '''
        sd = sd.SplitData(self.X,self.y,self.test_size,self.random_state)
        self.X_train,self.X_test,self.y_train,self.y_test = sd.split_data()

        """ We used estandarizacion class to standardize the data """
        st = st.Standardization(self.X_train,self.y_train)
        self.Xs,self.ys, self.yprom, self.ystd,self.Xprom,self.Xstd= st.standardize(self.X_train,self.y_train)


        ###############################################################################################

        ###############################################################################################

        """ We processing the data in pca method"""
        if numComponents>self.Xs.shape[1]: # Dont forget that Xs is X_train but standardized
            raise ValueError('numComponents must be less than the number of features')
        pca = PCA(numComponents)
        pca.fit(self.Xs)
        """ Transforming the data of the training set (Features) """
        self.score = pca.transform(self.Xs)

        """ We obtain the number of measurings of the features """
        m = self.score.shape[0]
        X_transformed_pca = np.concatenate((np.ones((m,1)), self.score), axis=1) # Here, we've added a column of ones to matrix 
                                                                                 # of features

        """ we define the number of features """
        n = X_transformed_pca.shape[1]

        """
        This apart we going to make the perceptron model
        """

        """ Variables for the perceptron model """
        self.w = np.zeros((n,1))
        self.MSE = np.zeros((m,1))
        self.wEpocas = np.zeros(n,numEpochs)

        """ Training """
        for i in range(numEpochs):
            # Calculating the gradiente 
            self.nablaMSE = 2@X_transformed_pca.T@(X_transformed_pca*self.w-self.ys)/m
            # we update the weights
            self.w = self.w - self.eta*self.nablaMSE
            # We save the current values of the weights
            self.wEpocas[:,i] = self.w
            # We determine the perfomance the our model
            ypred = X_transformed_pca@self.w@self.ystd + self.yprom
            # We determine the MSE in the training
            self.MSE = np.sum((self.X_train*self.w-self.y_train)**2)/m

