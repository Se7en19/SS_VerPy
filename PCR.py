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

# Configurar logging para debugging
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
        
        # Nuevas variables para debugging y rendimiento
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
        logger.info("Iniciando entrenamiento del modelo PCR")
        
        # Validaciones mejoradas con mensajes informativos
        self._validate_inputs(X, y, eta, numEpochs, test_size, random_state, numComponents)
        
        # Inicialización de variables
        init_start = time.time()
        self._initialize_variables(X, y, eta, numEpochs, test_size, random_state, numComponents)
        self.initialization_time = time.time() - init_start
        logger.info(f"Tiempo de inicialización: {self.initialization_time:.4f} segundos")

        # Limpieza de datos
        logger.info("Limpiando datos...")
        cleaner = cd.CleanData()
        self.X = cleaner.clean_nans(self.X)
        self.y = cleaner.clean_nans(self.y)
        logger.info(f"Datos limpios - X: {self.X.shape}, y: {self.y.shape}")

        # División de datos
        logger.info("Dividiendo datos en train/test...")
        splitter = sd.SplitData(self.X, self.y, self.test_size, self.random_state)
        self.X_train, self.X_test, self.y_train, self.y_test = splitter.split_data()
        logger.info(f"Train: {self.X_train.shape}, Test: {self.X_test.shape}")

        # Estandarización
        logger.info("Estandarizando datos...")
        standardizer = st.Standardization(self.X_train, self.y_train)
        self.Xs, self.ys, self.yprom, self.ystd, self.Xprom, self.Xstd = standardizer.standardize(self.X_train, self.y_train)
        logger.info("Estandarización completada")

        # PCA
        pca_start = time.time()
        logger.info(f"Aplicando PCA con {numComponents} componentes...")
        self._apply_pca(numComponents)
        self.pca_time = time.time() - pca_start
        logger.info(f"PCA completado en {self.pca_time:.4f} segundos")

        # Entrenamiento del perceptrón
        logger.info("Iniciando entrenamiento del perceptrón...")
        self._train_perceptron(numEpochs)
        
        self.training_time = time.time() - start_time
        logger.info(f"Entrenamiento completado en {self.training_time:.4f} segundos")
        logger.info(f"Tiempo total: {self.training_time:.4f} segundos")
        
        return (self.X_test, self.y_test, self.Xstd, self.Xprom, self.w, 
                self.ystd, self.yprom, self.coeff, self.score, self.latent, self.MSE)

    def _validate_inputs(self, X, y, eta, numEpochs, test_size, random_state, numComponents):
        """Validación mejorada de entradas"""
        logger.debug("Validando entradas...")
        
        # Validar X e y
        if X is None or y is None:
            raise ValueError('X e y no pueden ser None')
        
        # Convertir numpy arrays a DataFrames si es necesario
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
            logger.debug("X convertido de numpy array a DataFrame")
        if isinstance(y, np.ndarray):
            y = pd.DataFrame(y)
            logger.debug("y convertido de numpy array a DataFrame")
        
        # Validar tipos
        if not (isinstance(X, pd.DataFrame) and isinstance(y, pd.DataFrame)):
            raise TypeError('X e y deben ser DataFrames o numpy arrays')
        
        # Validar dimensiones
        if X.shape[0] != y.shape[0]:
            raise ValueError(f'X y y deben tener el mismo número de filas. X: {X.shape[0]}, y: {y.shape[0]}')
        
        # Validar parámetros numéricos
        if not (0 < eta < 1):
            raise ValueError(f'eta debe estar entre 0 y 1, recibido: {eta}')
        if numEpochs <= 0:
            raise ValueError(f'numEpochs debe ser positivo, recibido: {numEpochs}')
        if not (0 < test_size < 1):
            raise ValueError(f'test_size debe estar entre 0 y 1, recibido: {test_size}')
        if numComponents <= 0:
            raise ValueError(f'numComponents debe ser positivo, recibido: {numComponents}')
        
        logger.info("Validación de entradas completada exitosamente")

    def _initialize_variables(self, X, y, eta, numEpochs, test_size, random_state, numComponents):
        """Inicialización de variables de la clase"""
        self.X = X.copy()
        self.y = y.copy()
        self.eta = eta
        self.numEpochs = numEpochs
        self.test_size = test_size
        self.random_state = random_state
        
        # Log de parámetros
        logger.debug(f"Parámetros: eta={eta}, epochs={numEpochs}, test_size={test_size}, components={numComponents}")

    def _apply_pca(self, numComponents):
        """Aplicar PCA con validaciones"""
        if numComponents > self.Xs.shape[1]:
            raise ValueError(f'numComponents ({numComponents}) debe ser menor que el número de features ({self.Xs.shape[1]})')
        
        pca = PCA(n_components=numComponents)
        pca.fit(self.Xs)
        
        # Transformar datos de entrenamiento
        self.score = pca.transform(self.Xs)
        
        # Guardar resultados de PCA
        self.coeff = pca.components_
        self.latent = pca.explained_variance_
        
        logger.debug(f"PCA aplicado: {self.score.shape}, varianza explicada: {np.sum(pca.explained_variance_ratio_):.4f}")

    def _train_perceptron(self, numEpochs):
        """Entrenamiento del perceptrón con monitoreo de convergencia"""
        m = self.score.shape[0]
        n = self.score.shape[1] + 1  # +1 para el bias
        
        # Preparar matriz de features con bias
        X_transformed_pca = np.concatenate((np.ones((m, 1)), self.score), axis=1)
        
        # Inicializar pesos y métricas
        self.w = np.zeros((n, 1))
        self.MSE = np.zeros(numEpochs)
        self.wEpocas = np.zeros((n, numEpochs))
        
        # Monitoreo de convergencia
        best_mse = float('inf')
        patience_counter = 0
        patience = max(10, numEpochs // 10)  # 10% de epochs como patience
        
        logger.info(f"Iniciando entrenamiento: {m} muestras, {n} features")
        
        for i in range(numEpochs):
            # Forward pass y cálculo del gradiente
            y_pred = X_transformed_pca @ self.w
            error = y_pred - self.ys
            nablaMSE = 2 * X_transformed_pca.T @ error / m
            
            # Actualización de pesos
            self.w = self.w - self.eta * nablaMSE
            
            # Guardar pesos de la época
            self.wEpocas[:, i] = self.w.flatten()
            
            # Calcular MSE en escala original
            ypred_original = y_pred * self.ystd + self.yprom
            current_mse = np.sqrt(np.sum((ypred_original.flatten() - self.y_train.values.flatten())**2) / m)
            self.MSE[i] = current_mse
            
            # Monitoreo de convergencia
            if current_mse < best_mse:
                best_mse = current_mse
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Log cada 10% de epochs
            if (i + 1) % max(1, numEpochs // 10) == 0:
                logger.info(f"Época {i+1}/{numEpochs}: MSE = {current_mse:.6f}")
            
            # Early stopping si no hay mejora
            if patience_counter >= patience:
                logger.info(f"Early stopping en época {i+1} - No hay mejora en {patience} épocas")
                # Ajustar arrays al tamaño real
                self.MSE = self.MSE[:i+1]
                self.wEpocas = self.wEpocas[:, :i+1]
                break
        
        logger.info(f"Entrenamiento completado. Mejor MSE: {best_mse:.6f}")

    def get_performance_metrics(self):
        """Obtener métricas de rendimiento del modelo"""
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
        """Realizar predicciones con el modelo entrenado"""
        if not hasattr(self, 'w') or self.w is None:
            raise ValueError("El modelo debe ser entrenado antes de hacer predicciones")
        
        # Estandarizar nuevos datos
        X_new_std = (X_new - self.Xprom) / self.Xstd
        
        # Aplicar PCA
        X_new_pca = PCA(n_components=self.score.shape[1]).fit_transform(X_new_std)
        
        # Añadir bias
        X_new_with_bias = np.concatenate((np.ones((X_new_pca.shape[0], 1)), X_new_pca), axis=1)
        
        # Predicción
        y_pred_std = X_new_with_bias @ self.w
        y_pred = y_pred_std * self.ystd + self.yprom
        
        return y_pred

