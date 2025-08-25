#!/usr/bin/env python3
"""
Testing script to debug and measure the performance of the PCR model
"""

import numpy as np
import pandas as pd
import time
import cProfile
import pstats
from PCR import PCR

def create_test_data(n_samples=1000, n_features=10, noise_level=0.1):
    """Create synthetic test data"""
    print("Creating test data...")
    
    # Generate random features
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    
    # Create target with linear relationship + noise
    true_weights = np.random.randn(n_features)
    y = X @ true_weights + noise_level * np.random.randn(n_samples)
    
    # Convert to DataFrames
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    y_df = pd.DataFrame(y, columns=['target'])
    
    print(f" Data created: X={X_df.shape}, y={y_df.shape}")
    return X_df, y_df

def test_pcr_basic():
    """Basic test of the PCR model"""
    print("\n BASIC TEST OF THE PCR MODEL")
    print("=" * 50)
    
    # Create test data
    X, y = create_test_data(n_samples=500, n_features=5)
    
    # Create and train model
    pcr_model = PCR()
    
    try:
        print("\n Training model...")
        results = pcr_model.PCR(
            X=X, 
            y=y, 
            eta=0.01, 
            numEpochs=100, 
            test_size=0.2, 
            random_state=42, 
            numComponents=3
        )
        
        print("Training successful!")
        
        # Show performance metrics
        metrics = pcr_model.get_performance_metrics()
        print("\n PERFORMANCE METRICS:")
        print(f"   Total time: {metrics['training_time']:.4f} seconds")
        print(f"   PCA time: {metrics['pca_time']:.4f} seconds")
        print(f"   Initialization time: {metrics['initialization_time']:.4f} seconds")
        print(f"   Final MSE: {metrics['final_mse']:.6f}")
        print(f"   Best MSE: {metrics['best_mse']:.6f}")
        print(f"   Convergence epochs: {metrics['convergence_epochs']}")
        
        # Show results
        print(f"\n RESULTS:")
        print(f"   PCA coefficients: {pcr_model.coeff.shape}")
        print(f"   Explained variance: {pcr_model.latent}")
        print(f"   Final weights: {pcr_model.w.shape}")
        
        return True
        
    except Exception as e:
        print(f" Error during training: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pcr_edge_cases():
    """Edge cases and validations test"""
    print("\n EDGE CASES TEST")
    print("=" * 50)
    
    pcr_model = PCR()
    
    # Test 1: None data
    print("\n Test 1: None data")
    try:
        pcr_model.PCR(None, None, 0.01, 100, 0.2, 42, 3)
        print(" It should have failed with None data")
    except ValueError as e:
        print(f" Correctly captured: {e}")
    
    # Test 2: Invalid parameters
    print("\n Test 2: Invalid parameters")
    X, y = create_test_data(100, 5)
    
    try:
        pcr_model.PCR(X, y, -0.01, 100, 0.2, 42, 3)  # negative eta
        print(" It should have failed with negative eta")
    except ValueError as e:
        print(f" Correctly captured: {e}")
    
    try:
        pcr_model.PCR(X, y, 0.01, 0, 0.2, 42, 3)  # epochs = 0
        print(" It should have failed with epochs = 0")
    except ValueError as e:
        print(f" Correctly captured: {e}")
    
    try:
        pcr_model.PCR(X, y, 0.01, 100, 1.5, 42, 3)  # test_size > 1
        print("It should have failed with test_size > 1")
    except ValueError as e:
        print(f"Correctly captured: {e}")

def test_pcr_performance():
    """Performance test with different data sizes"""
    print("\n PERFORMANCE TEST")
    print("=" * 50)
    
    sizes = [100, 500, 1000]
    
    for size in sizes:
        print(f"\n Testing with {size} samples...")
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
            
            print(f"   Total time: {total_time:.4f}s")
            print(f"   Final MSE: {metrics['final_mse']:.6f}")
            print(f"   Epochs: {metrics['convergence_epochs']}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")

def profile_pcr():
    """Profiling of the PCR model"""
    print("\n PCR MODEL PROFILING")
    print("=" * 50)
    
    X, y = create_test_data(n_samples=1000, n_features=10)
    pcr_model = PCR()
    
    # Create profiler
    profiler = cProfile.Profile()
    profiler.enable()
    
    try:
        results = pcr_model.PCR(
            X=X, y=y, eta=0.01, numEpochs=100, 
            test_size=0.2, random_state=42, numComponents=5
        )
        
        profiler.disable()
        
        # Show statistics
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        
        print(" TOP 10 SLOWEST FUNCTIONS:")
        stats.print_stats(10)
        
        print("\n STATS BY FILE:")
        stats.print_stats('PCR.py')
        
    except Exception as e:
        print(f" Error during profiling: {e}")
        profiler.disable()

def main():
    """Main testing function"""
    print(" STARTING PCR MODEL TESTS")
    print("=" * 60)
    
    # Basic test
    basic_success = test_pcr_basic()
    
    if basic_success:
        # Edge cases test
        test_pcr_edge_cases()
        
        # Performance test
        test_pcr_performance()
        
        # Profiling
        profile_pcr()
        
        print("\n ALL TESTS COMPLETED!")
    else:
        print("\n Basic test failed. Check the errors before continuing.")

if __name__ == "__main__":
    main()
