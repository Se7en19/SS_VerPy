"""
This script is a regression model, using the PCR algorithm and my own perceptron algorithm.
by: Se7en19 
"""

""" Importing libraries """
import pandas as pd
import numpy as np
import time
import logging
from pandas.core.base import NoNewAttributesMixin
import CleanData as cd 
import split_data as sd
import estandarizacion as st 
from sklearn.decomposition import PCA

# Configure logging for debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
        self.numComponents = None
        
        self.training_time = None
        self.pca_time = None
        self.initialization_time = None
        
    def PCR(self, X, y, eta, numEpochs, test_size, random_state, numComponents):
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
        numComponents: int with the number of PCA components.

        Returns
        X_test, y_test, Xstd, Xprom, w, ystd, yprom, coeff, score, latent, MSE
        
        """
        start_time = time.time()
        logger.info("Starting PCR model training")
        
        # Improved validations with informative messages
        self._validate_inputs(X, y, eta, numEpochs, test_size, random_state, numComponents)
        
        # Variables initialization
        init_start = time.time()
        self._initialize_variables(X, y, eta, numEpochs, test_size, random_state, numComponents)
        self.initialization_time = time.time() - init_start
        logger.info(f"Initialization time: {self.initialization_time:.4f} seconds")

        # Data cleaning
        # self.X = X.copy()
        # self.y = y.copy()
        logger.info("Cleaning data...")
        cleaner = cd.CleanData()
        self.X = cleaner.clean_nans(self.X)
        self.y = cleaner.clean_nans(self.y)
        logger.info(f"Clean data - X: {self.X.shape}, y: {self.y.shape}")

        # Train/test split
        logger.info("Splitting data into train/test...")
        splitter = sd.SplitData(self.X, self.y, self.test_size, self.random_state)
        self.X_train, self.X_test, self.y_train, self.y_test = splitter.split_data()
        logger.info(f"Train: {self.X_train.shape}, Test: {self.X_test.shape}")

        # Standardization
        logger.info("Standardizing data...")
        standardizer = st.Standardization()
        self.Xs, self.ys, self.yprom, self.ystd, self.Xprom, self.Xstd = standardizer.standardize(self.X_train, self.y_train)
        logger.info("Standardization completed")

        # PCA
        pca_start = time.time()
        logger.info(f"Applying PCA with {numComponents} components...")
        self._apply_pca(numComponents)
        self.pca_time = time.time() - pca_start
        logger.info(f"PCA completed in {self.pca_time:.4f} seconds")

        # Perceptron training
        logger.info("Starting perceptron training...")
        self._train_perceptron(numEpochs)
        
        self.training_time = time.time() - start_time
        logger.info(f"Training completed in {self.training_time:.4f} seconds")
        logger.info(f"Total time: {self.training_time:.4f} seconds")
        
        return (self.X_test, self.y_test, self.Xstd, self.Xprom, self.w, 
                self.ystd, self.yprom, self.coeff, self.score, self.latent, self.MSE)

    def _validate_inputs(self, X, y, eta, numEpochs, test_size, random_state, numComponents):
        """Improved input validation"""
        logger.debug("Validating inputs...")
        
        # Validate X and y
        if X is None or y is None:
            raise TypeError('X,y can\'t be None')
        
        # Validate types
        valid_types = (pd.DataFrame, np.ndarray)

        if not isinstance(X, valid_types):
            raise TypeError("X debe ser pandas.DataFrame o numpy.ndarray")

        if not isinstance(y, valid_types):
            raise TypeError("y debe ser pandas.DataFrame o numpy.ndarray")
        
        # Validate shapes
        if X.shape[0] > y.shape[0]:
            # we obtain the number of measuring of y
            m = y.shape[0]
            X = X.iloc[ :m , : ]
            return X
        elif X.shape[0] < y.shape[0]:
            # we obtain the number of measuring of X
            m = X.shape[0]
            y = y.iloc[ :m , : ]
            return y


        # Validate numeric parameters
        if not (0 < eta < 1):
            raise ValueError(f'eta must be between 0 and 1, received: {eta}')
        if numEpochs <= 0:
            raise ValueError(f'numEpochs must be positive, received: {numEpochs}')
        if not (0 < test_size < 1):
            raise ValueError(f'test_size must be between 0 and 1, received: {test_size}')
        if numComponents <= 0:
            raise ValueError(f'numComponents must be positive, received: {numComponents}')
        
        logger.info("Input validation completed successfully")
        

    def _initialize_variables(self, X, y, eta, numEpochs, test_size, random_state, numComponents):
        """Class variables initialization"""
        self.X = X.copy()
        self.y = y.copy()
        self.eta = eta
        self.numEpochs = numEpochs
        self.test_size = test_size
        self.random_state = random_state
        
        # Parameters log
        logger.debug(f"Parameters: eta={eta}, epochs={numEpochs}, test_size={test_size}, components={numComponents}")

    def _apply_pca(self, numComponents):
        ''' we transform Xs (dataframe) to numpy array'''
        self.Xs = self.Xs.values
        """Apply PCA with validations"""
        if numComponents > self.Xs.shape[1]:
            raise ValueError(f'numComponents ({numComponents}) must be less than the number of features ({self.Xs.shape[1]})')
        
        pca = PCA(n_components=numComponents)
        pca.fit(self.Xs)
        
        # Transform training data
        self.score = pca.transform(self.Xs)
        
        # Save PCA results
        self.coeff = pca.components_
        self.latent = pca.explained_variance_
        
        logger.debug(f"PCA applied: {self.score.shape}, explained variance: {np.sum(pca.explained_variance_ratio_):.4f}")

    def _train_perceptron(self, numEpochs):
        # Transform ys (DataFrame) to ndarray 
        self.ys = self.ys.values
        self.y_train = self.y_train.values
        """Perceptron training with convergence monitoring"""
        m = self.score.shape[0]
        n = self.score.shape[1] + 1  # +1 for bias
        
        # Prepare features matrix with bias
        X_transformed_pca = np.concatenate((np.ones((m, 1)), self.score), axis=1)
        
        # Initialize weights and metrics
        self.w = np.zeros((n, 1))
        self.MSE = np.zeros(numEpochs)
        self.wEpocas = np.zeros((n, numEpochs))
        
        # Convergence monitoring
        best_mse = float('inf')
        patience_counter = 0
        patience = max(10, numEpochs // 10)  # 10% of epochs as patience
        
        logger.info(f"Starting training: {m} samples, {n} features")
        
        for i in range(numEpochs):
            # Forward pass and gradient calculation
            y_pred = X_transformed_pca @ self.w
            error = y_pred - self.ys
            nablaMSE = 2 * X_transformed_pca.T @ error / m
            
            # Weights update
            self.w = self.w - self.eta * nablaMSE
            
            # Save epoch weights
            self.wEpocas[:, i] = self.w.flatten()
            
            # Compute MSE in original scale
            ypred_original = y_pred * self.ystd + self.yprom
            current_mse = np.sqrt(np.sum((ypred_original.flatten() - self.y_train.flatten())**2) / m)
            self.MSE[i] = current_mse
            
            # Convergence monitoring
            if current_mse < best_mse:
                best_mse = current_mse
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Log every 10% of epochs
            if (i + 1) % max(1, numEpochs // 10) == 0:
                logger.info(f"Epoch {i+1}/{numEpochs}: MSE = {current_mse:.6f}")
            
            # Early stopping if there is no improvement
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {i+1} - No improvement in {patience} epochs")
                # Adjust arrays to the actual size
                self.MSE = self.MSE[:i+1]
                self.wEpocas = self.wEpocas[:, :i+1]
                break
        
        logger.info(f"Training completed. Best MSE: {best_mse:.6f}")

    def get_performance_metrics(self):
        """Get model performance metrics"""
        return {
            'training_time': self.training_time,
            'pca_time': self.pca_time,
            'initialization_time': self.initialization_time,
            'final_mse': self.MSE[-1] if len(self.MSE) > 0 else None,
            'best_mse': np.min(self.MSE) if len(self.MSE) > 0 else None,
            'convergence_epochs': len(self.MSE),
            'total_epochs': self.numEpochs
        }

    def predict(self, X_new):
        """Make predictions with the trained model"""
        if not hasattr(self, 'w') or self.w is None:
            raise ValueError("The model must be trained before making predictions")
        
        # Standardize new data
        X_new_std = (X_new - self.Xprom) / self.Xstd
        
        # Apply PCA
        X_new_pca = PCA(n_components=self.score.shape[1]).fit_transform(X_new_std)
        
        # Add bias
        X_new_with_bias = np.concatenate((np.ones((X_new_pca.shape[0], 1)), X_new_pca), axis=1)
        
        # Prediction
        y_pred_std = X_new_with_bias @ self.w
        y_pred = y_pred_std * self.ystd + self.yprom
        
        return y_pred

