#!/usr/bin/env python3
"""
Script de testing para debuggear y medir el rendimiento del modelo PCR
"""

import numpy as np
import pandas as pd
import time
import cProfile
import pstats
from PCR import PCR

def create_test_data(n_samples=1000, n_features=10, noise_level=0.1):
    """Crear datos de prueba sintéticos"""
    print("🔧 Creando datos de prueba...")
    
    # Generar features aleatorias
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    
    # Crear target con relación lineal + ruido
    true_weights = np.random.randn(n_features)
    y = X @ true_weights + noise_level * np.random.randn(n_samples)
    
    # Convertir a DataFrames
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    y_df = pd.DataFrame(y, columns=['target'])
    
    print(f"✅ Datos creados: X={X_df.shape}, y={y_df.shape}")
    return X_df, y_df

def test_pcr_basic():
    """Test básico del modelo PCR"""
    print("\n🧪 TEST BÁSICO DEL MODELO PCR")
    print("=" * 50)
    
    # Crear datos de prueba
    X, y = create_test_data(n_samples=500, n_features=5)
    
    # Crear y entrenar modelo
    pcr_model = PCR()
    
    try:
        print("\n🚀 Entrenando modelo...")
        results = pcr_model.PCR(
            X=X, 
            y=y, 
            eta=0.01, 
            numEpochs=100, 
            test_size=0.2, 
            random_state=42, 
            numComponents=3
        )
        
        print("✅ Entrenamiento exitoso!")
        
        # Mostrar métricas de rendimiento
        metrics = pcr_model.get_performance_metrics()
        print("\n📊 MÉTRICAS DE RENDIMIENTO:")
        print(f"   Tiempo total: {metrics['training_time']:.4f} segundos")
        print(f"   Tiempo PCA: {metrics['pca_time']:.4f} segundos")
        print(f"   Tiempo inicialización: {metrics['initialization_time']:.4f} segundos")
        print(f"   MSE final: {metrics['final_mse']:.6f}")
        print(f"   Mejor MSE: {metrics['best_mse']:.6f}")
        print(f"   Épocas de convergencia: {metrics['convergence_epochs']}")
        
        # Mostrar resultados
        print(f"\n📈 RESULTADOS:")
        print(f"   Coeficientes PCA: {pcr_model.coeff.shape}")
        print(f"   Varianza explicada: {pcr_model.latent}")
        print(f"   Pesos finales: {pcr_model.w.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pcr_edge_cases():
    """Test de casos límite y validaciones"""
    print("\n🧪 TEST DE CASOS LÍMITE")
    print("=" * 50)
    
    pcr_model = PCR()
    
    # Test 1: Datos None
    print("\n🔍 Test 1: Datos None")
    try:
        pcr_model.PCR(None, None, 0.01, 100, 0.2, 42, 3)
        print("❌ Debería haber fallado con datos None")
    except ValueError as e:
        print(f"✅ Correctamente capturado: {e}")
    
    # Test 2: Parámetros inválidos
    print("\n🔍 Test 2: Parámetros inválidos")
    X, y = create_test_data(100, 5)
    
    try:
        pcr_model.PCR(X, y, -0.01, 100, 0.2, 42, 3)  # eta negativo
        print("❌ Debería haber fallado con eta negativo")
    except ValueError as e:
        print(f"✅ Correctamente capturado: {e}")
    
    try:
        pcr_model.PCR(X, y, 0.01, 0, 0.2, 42, 3)  # epochs = 0
        print("❌ Debería haber fallado con epochs = 0")
    except ValueError as e:
        print(f"✅ Correctamente capturado: {e}")
    
    try:
        pcr_model.PCR(X, y, 0.01, 100, 1.5, 42, 3)  # test_size > 1
        print("❌ Debería haber fallado con test_size > 1")
    except ValueError as e:
        print(f"✅ Correctamente capturado: {e}")

def test_pcr_performance():
    """Test de rendimiento con diferentes tamaños de datos"""
    print("\n🧪 TEST DE RENDIMIENTO")
    print("=" * 50)
    
    sizes = [100, 500, 1000]
    
    for size in sizes:
        print(f"\n📊 Probando con {size} muestras...")
        X, y = create_test_data(n_samples=size, n_features=10)
        
        pcr_model = PCR()
        start_time = time.time()
        
        try:
            results = pcr_model.PCR(
                X=X, y=y, eta=0.01, numEpochs=50, 
                test_size=0.2, random_state=42, numComponents=5
            )
            
            total_time = time.time() - start_time
            metrics = pcr_model.get_performance_metrics()
            
            print(f"   ✅ Tiempo total: {total_time:.4f}s")
            print(f"   ✅ MSE final: {metrics['final_mse']:.6f}")
            print(f"   ✅ Épocas: {metrics['convergence_epochs']}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")

def profile_pcr():
    """Profiling del modelo PCR"""
    print("\n🧪 PROFILING DEL MODELO PCR")
    print("=" * 50)
    
    X, y = create_test_data(n_samples=1000, n_features=10)
    pcr_model = PCR()
    
    # Crear profiler
    profiler = cProfile.Profile()
    profiler.enable()
    
    try:
        results = pcr_model.PCR(
            X=X, y=y, eta=0.01, numEpochs=100, 
            test_size=0.2, random_state=42, numComponents=5
        )
        
        profiler.disable()
        
        # Mostrar estadísticas
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        
        print("📊 TOP 10 FUNCIONES MÁS LENTAS:")
        stats.print_stats(10)
        
        print("\n📊 ESTADÍSTICAS POR ARCHIVO:")
        stats.print_stats('PCR.py')
        
    except Exception as e:
        print(f"❌ Error durante profiling: {e}")
        profiler.disable()

def main():
    """Función principal de testing"""
    print("🚀 INICIANDO TESTS DEL MODELO PCR")
    print("=" * 60)
    
    # Test básico
    basic_success = test_pcr_basic()
    
    if basic_success:
        # Test de casos límite
        test_pcr_edge_cases()
        
        # Test de rendimiento
        test_pcr_performance()
        
        # Profiling
        profile_pcr()
        
        print("\n🎉 TODOS LOS TESTS COMPLETADOS!")
    else:
        print("\n❌ El test básico falló. Revisa los errores antes de continuar.")

if __name__ == "__main__":
    main()
