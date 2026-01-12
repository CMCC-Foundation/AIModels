'''
TimeSeriesTransformer Example Usage
==================================

This script demonstrates how to use the TimeSeriesTransformer model for various
time series prediction tasks. It includes examples for:

1. Basic usage with synthetic data
2. Real-world data loading and preprocessing
3. Different model configurations
4. Various loss functions
5. Training with different parameters
6. Greedy inference for long sequences
7. Model evaluation and visualization

'''

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
from TimeSeriesTransformer import (
    TimeSeriesTransformer, 
    TimeSeriesDataset, 
    TimeSeriesTrainer, 
    TimeSeriesPredictor,
    create_data_loaders,
    LossFunctions
)
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')


def generate_synthetic_data(L=1000, M=5, noise_level=0.1, seed=42):
    """
    Generate synthetic time series data with multiple patterns.
    
    Parameters:
    -----------
    L : int
        Number of time steps
    M : int
        Number of features
    noise_level : float
        Level of noise to add
    seed : int
        Random seed
        
    Returns:
    --------
    data : np.ndarray
        Generated time series data
    """
    np.random.seed(seed)
    
    # Time axis
    t = np.linspace(0, 4*np.pi, L)
    
    # Generate different patterns for each feature
    data = np.zeros((L, M))
    
    # Feature 0: Sine wave
    data[:, 0] = np.sin(t)
    
    # Feature 1: Cosine wave with different frequency
    data[:, 1] = np.cos(2*t)
    
    # Feature 2: Trend + seasonal
    data[:, 2] = 0.1*t + np.sin(3*t) + 0.5*np.cos(t)
    
    # Feature 3: Random walk
    data[:, 3] = np.cumsum(np.random.randn(L)) * 0.1
    
    # Feature 4: Step function
    data[:, 4] = np.where(t < 2*np.pi, 1, -1)
    
    # Add noise
    data += np.random.randn(L, M) * noise_level
    
    return data


def load_air_passengers_data():
    """
    Load and preprocess the Air Passengers dataset.
    
    Returns:
    --------
    data : np.ndarray
        Preprocessed time series data
    """
    try:
        # Try to load from pandas
        df = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv')
        data = df['Passengers'].values.reshape(-1, 1)
        
        # Add some derived features
        data = np.column_stack([
            data,  # Original data
            np.roll(data, 1),  # Lagged values
            np.roll(data, 12),  # 12-month lag
            np.arange(len(data)),  # Time trend
            np.sin(2*np.pi*np.arange(len(data))/12)  # Seasonal component
        ])
        
        # Remove NaN values
        data = data[12:, :]
        
        return data
        
    except:
        print("Could not load Air Passengers data, using synthetic data instead")
        return generate_synthetic_data(L=500, M=5)


def example_basic_usage():
    """
    Basic usage example with synthetic data.
    """
    print("=" * 50)
    print("BASIC USAGE EXAMPLE")
    print("=" * 50)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Generate data
    data = generate_synthetic_data(L=1000, M=5)
    
    # Normalize data
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data)
    
    # Model parameters
    TIN = 50  # Input sequence length
    T = 10    # Output sequence length
    M = data.shape[1]  # Number of input features
    N = M     # Number of output features
    
    print(f"Data shape: {data.shape}")
    print(f"Input sequence length (TIN): {TIN}")
    print(f"Output sequence length (T): {T}")
    print(f"Input features (M): {M}")
    print(f"Output features (N): {N}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        data_normalized, TIN, T, batch_size=32
    )
    
    # Create model
    model = TimeSeriesTransformer(
        input_size=M,
        output_size=N,
        d_model=128,
        n_heads=8,
        n_layers=4,
        d_ff=512,
        dropout=0.1
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer and train
    trainer = TimeSeriesTrainer(model, device)
    
    train_losses, val_losses = trainer.train(
        train_loader, val_loader,
        epochs=30,
        lr=0.001,
        patience=5,
        loss_type='mse'
    )
    
    # Evaluate model
    predictor = TimeSeriesPredictor(model, device)
    results, predictions, targets = predictor.evaluate(test_loader)
    
    print("\nTest Results:")
    for metric, value in results.items():
        print(f"{metric.upper()}: {value:.6f}")
    
    # Plot training history
    trainer.plot_training_history()
    
    # Plot predictions
    predictor.plot_predictions(predictions, targets, sample_idx=0, feature_idx=0)
    
    return model, predictor, results


def example_different_losses():
    """
    Example with different loss functions.
    """
    print("\n" + "=" * 50)
    print("DIFFERENT LOSS FUNCTIONS EXAMPLE")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Generate data
    data = generate_synthetic_data(L=800, M=3)
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data)
    
    # Model parameters
    TIN, T = 40, 8
    M = data.shape[1]
    N = M
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        data_normalized, TIN, T, batch_size=16
    )
    
    # Test different loss functions
    loss_functions = ['mse', 'mae', 'huber', 'custom']
    results_comparison = {}
    
    for loss_type in loss_functions:
        print(f"\nTraining with {loss_type.upper()} loss...")
        
        # Create model
        model = TimeSeriesTransformer(
            input_size=M,
            output_size=N,
            d_model=64,
            n_heads=4,
            n_layers=3,
            d_ff=256,
            dropout=0.1
        )
        
        # Train model
        trainer = TimeSeriesTrainer(model, device)
        train_losses, val_losses = trainer.train(
            train_loader, val_loader,
            epochs=20,
            lr=0.001,
            patience=5,
            loss_type=loss_type,
            alpha=0.7 if loss_type == 'custom' else 0.5
        )
        
        # Evaluate
        predictor = TimeSeriesPredictor(model, device)
        results, _, _ = predictor.evaluate(test_loader)
        results_comparison[loss_type] = results
    
    # Compare results
    print("\nLoss Function Comparison:")
    print("-" * 40)
    for loss_type, results in results_comparison.items():
        print(f"{loss_type.upper():8s}: MSE={results['mse']:.6f}, MAE={results['mae']:.6f}")


def example_model_configurations():
    """
    Example with different model configurations.
    """
    print("\n" + "=" * 50)
    print("DIFFERENT MODEL CONFIGURATIONS EXAMPLE")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Generate data
    data = generate_synthetic_data(L=600, M=4)
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data)
    
    # Model parameters
    TIN, T = 30, 5
    M = data.shape[1]
    N = M
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        data_normalized, TIN, T, batch_size=16
    )
    
    # Different model configurations
    configurations = [
        {
            'name': 'Small',
            'd_model': 64,
            'n_heads': 4,
            'n_layers': 2,
            'd_ff': 256
        },
        {
            'name': 'Medium',
            'd_model': 128,
            'n_heads': 8,
            'n_layers': 4,
            'd_ff': 512
        },
        {
            'name': 'Large',
            'd_model': 256,
            'n_heads': 16,
            'n_layers': 6,
            'd_ff': 1024
        }
    ]
    
    results_comparison = {}
    
    for config in configurations:
        print(f"\nTraining {config['name']} model...")
        
        # Create model
        model = TimeSeriesTransformer(
            input_size=M,
            output_size=N,
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            n_layers=config['n_layers'],
            d_ff=config['d_ff'],
            dropout=0.1
        )
        
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Train model
        trainer = TimeSeriesTrainer(model, device)
        train_losses, val_losses = trainer.train(
            train_loader, val_loader,
            epochs=15,
            lr=0.001,
            patience=3,
            loss_type='mse'
        )
        
        # Evaluate
        predictor = TimeSeriesPredictor(model, device)
        results, _, _ = predictor.evaluate(test_loader)
        results_comparison[config['name']] = results
    
    # Compare results
    print("\nModel Configuration Comparison:")
    print("-" * 40)
    for name, results in results_comparison.items():
        print(f"{name:8s}: MSE={results['mse']:.6f}, MAE={results['mae']:.6f}")


def example_greedy_inference():
    """
    Example demonstrating greedy inference for long sequences.
    """
    print("\n" + "=" * 50)
    print("GREEDY INFERENCE EXAMPLE")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Generate data
    data = generate_synthetic_data(L=500, M=3)
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data)
    
    # Model parameters
    TIN, T = 20, 5
    M = data.shape[1]
    N = M
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        data_normalized, TIN, T, batch_size=16
    )
    
    # Create and train model
    model = TimeSeriesTransformer(
        input_size=M,
        output_size=N,
        d_model=64,
        n_heads=4,
        n_layers=3,
        d_ff=256,
        dropout=0.1
    )
    
    trainer = TimeSeriesTrainer(model, device)
    trainer.train(train_loader, val_loader, epochs=20, lr=0.001, patience=5)
    
    # Test greedy inference
    predictor = TimeSeriesPredictor(model, device)
    
    # Take a sample input
    sample_input = torch.FloatTensor(data_normalized[:TIN]).unsqueeze(0)
    
    # Predict different lengths
    prediction_lengths = [5, 10, 20, 50]
    
    plt.figure(figsize=(15, 10))
    
    for i, T_predict in enumerate(prediction_lengths):
        print(f"Predicting {T_predict} time steps...")
        
        # Use greedy inference
        predictions = predictor.greedy_inference(sample_input, T_predict)
        
        # Plot results
        plt.subplot(2, 2, i+1)
        
        # Plot input sequence
        input_series = sample_input[0, :, 0].cpu().numpy()
        plt.plot(range(TIN), input_series, 'b-', label='Input', linewidth=2)
        
        # Plot predictions
        pred_series = predictions[0, :, 0].cpu().numpy()
        plt.plot(range(TIN, TIN + T_predict), pred_series, 'r--', label='Predictions', linewidth=2)
        
        plt.xlabel('Time Steps')
        plt.ylabel('Value')
        plt.title(f'Greedy Inference: {T_predict} Steps')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print("Greedy inference completed!")


def example_real_world_data():
    """
    Example with real-world data (Air Passengers dataset).
    """
    print("\n" + "=" * 50)
    print("REAL-WORLD DATA EXAMPLE")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load real-world data
    data = load_air_passengers_data()
    
    # Normalize data
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data)
    
    # Model parameters
    TIN, T = 24, 12  # 2 years input, 1 year prediction
    M = data.shape[1]
    N = 1  # Predict only the main feature
    
    print(f"Data shape: {data.shape}")
    print(f"Input sequence length (TIN): {TIN}")
    print(f"Output sequence length (T): {T}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        data_normalized, TIN, T, batch_size=8
    )
    
    # Create model
    model = TimeSeriesTransformer(
        input_size=M,
        output_size=N,
        d_model=128,
        n_heads=8,
        n_layers=4,
        d_ff=512,
        dropout=0.1
    )
    
    # Train model
    trainer = TimeSeriesTrainer(model, device)
    train_losses, val_losses = trainer.train(
        train_loader, val_loader,
        epochs=50,
        lr=0.001,
        patience=10,
        loss_type='mse'
    )
    
    # Evaluate model
    predictor = TimeSeriesPredictor(model, device)
    results, predictions, targets = predictor.evaluate(test_loader)
    
    print("\nTest Results:")
    for metric, value in results.items():
        print(f"{metric.upper()}: {value:.6f}")
    
    # Plot predictions
    plt.figure(figsize=(15, 5))
    
    # Plot training history
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    
    # Plot predictions vs targets
    plt.subplot(1, 2, 2)
    pred_series = predictions[0, :, 0]
    target_series = targets[0, :, 0]
    plt.plot(target_series, label='True', linewidth=2)
    plt.plot(pred_series, label='Predicted', linewidth=2, linestyle='--')
    plt.xlabel('Time Steps')
    plt.ylabel('Normalized Value')
    plt.title('Predictions vs Targets')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


def main():
    """
    Run all examples.
    """
    print("TimeSeriesTransformer Examples")
    print("=" * 60)
    
    # Run examples
    example_basic_usage()
    example_different_losses()
    example_model_configurations()
    example_greedy_inference()
    example_real_world_data()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
