'''
Test script for TimeSeriesTransformer
====================================

This script tests the basic functionality of the TimeSeriesTransformer model
to ensure everything works correctly.
'''

import torch
import numpy as np
from TimeSeriesTransformer import (
    TimeSeriesTransformer,
    TimeSeriesDataset,
    TimeSeriesTrainer,
    TimeSeriesPredictor,
    create_data_loaders
)
from sklearn.preprocessing import StandardScaler


def test_model_creation():
    """Test model creation and forward pass."""
    print("Testing model creation...")
    
    # Create model
    model = TimeSeriesTransformer(
        input_size=5,
        output_size=3,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=256,
        dropout=0.1
    )
    
    # Test forward pass
    batch_size, seq_len, input_features = 4, 20, 5
    x = torch.randn(batch_size, seq_len, input_features)
    
    with torch.no_grad():
        output = model(x)
    
    expected_shape = (batch_size, seq_len, 3)  # output_size = 3
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    
    print(f"✓ Model creation and forward pass successful")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    
    return model


def test_dataset():
    """Test dataset creation and loading."""
    print("\nTesting dataset creation...")
    
    # Generate sample data
    data = np.random.randn(100, 5)
    TIN, T = 10, 5
    
    # Create dataset
    dataset = TimeSeriesDataset(data, TIN, T)
    
    # Test dataset length
    expected_length = len(data) - TIN - T + 1
    assert len(dataset) == expected_length, f"Expected {expected_length}, got {len(dataset)}"
    
    # Test getting an item
    input_seq, target_seq = dataset[0]
    assert input_seq.shape == (TIN, 5), f"Expected input shape {(TIN, 5)}, got {input_seq.shape}"
    assert target_seq.shape == (T, 5), f"Expected target shape {(T, 5)}, got {target_seq.shape}"
    
    print(f"✓ Dataset creation successful")
    print(f"  Dataset length: {len(dataset)}")
    print(f"  Input sequence shape: {input_seq.shape}")
    print(f"  Target sequence shape: {target_seq.shape}")
    
    return dataset


def test_data_loaders():
    """Test data loader creation."""
    print("\nTesting data loader creation...")
    
    # Generate sample data
    data = np.random.randn(200, 4)
    TIN, T = 15, 8
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        data, TIN, T, batch_size=8
    )
    
    # Test data loaders
    for name, loader in [("Train", train_loader), ("Val", val_loader), ("Test", test_loader)]:
        batch = next(iter(loader))
        input_seq, target_seq = batch
        assert input_seq.shape[1] == TIN, f"{name} loader input shape error"
        assert target_seq.shape[1] == T, f"{name} loader target shape error"
        print(f"  {name} loader: {len(loader)} batches, batch shape: {input_seq.shape}")
    
    print(f"✓ Data loader creation successful")
    
    return train_loader, val_loader, test_loader


def test_training():
    """Test training functionality."""
    print("\nTesting training functionality...")
    
    # Generate small dataset for quick testing
    data = np.random.randn(100, 3)
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data)
    
    TIN, T = 10, 5
    train_loader, val_loader, test_loader = create_data_loaders(
        data_normalized, TIN, T, batch_size=4
    )
    
    # Create small model
    model = TimeSeriesTransformer(
        input_size=3,
        output_size=3,
        d_model=32,
        n_heads=2,
        n_layers=2,
        d_ff=128,
        dropout=0.1
    )
    
    # Create trainer
    device = torch.device('cpu')  # Use CPU for testing
    trainer = TimeSeriesTrainer(model, device)
    
    # Train for a few epochs
    train_losses, val_losses = trainer.train(
        train_loader, val_loader,
        epochs=3,
        lr=0.01,
        patience=2,
        loss_type='mse'
    )
    
    assert len(train_losses) > 0, "Training should produce loss values"
    assert len(val_losses) > 0, "Validation should produce loss values"
    
    print(f"✓ Training successful")
    print(f"  Training epochs: {len(train_losses)}")
    print(f"  Final training loss: {train_losses[-1]:.6f}")
    print(f"  Final validation loss: {val_losses[-1]:.6f}")
    
    return model, trainer


def test_prediction():
    """Test prediction functionality."""
    print("\nTesting prediction functionality...")
    
    # Generate test data
    data = np.random.randn(50, 3)
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data)
    
    # Create model
    model = TimeSeriesTransformer(
        input_size=3,
        output_size=3,
        d_model=32,
        n_heads=2,
        n_layers=2,
        d_ff=128,
        dropout=0.1
    )
    
    # Create predictor
    device = torch.device('cpu')
    predictor = TimeSeriesPredictor(model, device)
    
    # Test basic prediction
    input_seq = torch.FloatTensor(data_normalized[:10]).unsqueeze(0)  # (1, 10, 3)
    predictions = predictor.predict(input_seq, T_predict=5)
    
    expected_shape = (1, 5, 3)
    assert predictions.shape == expected_shape, f"Expected {expected_shape}, got {predictions.shape}"
    
    # Test greedy inference
    greedy_predictions = predictor.greedy_inference(input_seq, T_predict=8)
    expected_greedy_shape = (1, 8, 3)
    assert greedy_predictions.shape == expected_greedy_shape, f"Expected {expected_greedy_shape}, got {greedy_predictions.shape}"
    
    print(f"✓ Prediction successful")
    print(f"  Basic prediction shape: {predictions.shape}")
    print(f"  Greedy prediction shape: {greedy_predictions.shape}")
    
    return predictor


def test_loss_functions():
    """Test different loss functions."""
    print("\nTesting loss functions...")
    
    from TimeSeriesTransformer import LossFunctions
    
    # Create sample predictions and targets
    pred = torch.randn(4, 5, 3)
    target = torch.randn(4, 5, 3)
    
    # Test MSE loss
    mse_loss = LossFunctions.mse_loss(pred, target)
    assert mse_loss.item() > 0, "MSE loss should be positive"
    
    # Test MAE loss
    mae_loss = LossFunctions.mae_loss(pred, target)
    assert mae_loss.item() > 0, "MAE loss should be positive"
    
    # Test Huber loss
    huber_loss = LossFunctions.huber_loss(pred, target)
    assert huber_loss.item() > 0, "Huber loss should be positive"
    
    # Test custom loss
    custom_loss = LossFunctions.custom_loss(pred, target, alpha=0.7)
    assert custom_loss.item() > 0, "Custom loss should be positive"
    
    print(f"✓ All loss functions working")
    print(f"  MSE: {mse_loss.item():.6f}")
    print(f"  MAE: {mae_loss.item():.6f}")
    print(f"  Huber: {huber_loss.item():.6f}")
    print(f"  Custom: {custom_loss.item():.6f}")


def main():
    """Run all tests."""
    print("TimeSeriesTransformer Test Suite")
    print("=" * 40)
    
    try:
        # Run tests
        test_model_creation()
        test_dataset()
        test_data_loaders()
        test_training()
        test_prediction()
        test_loss_functions()
        
        print("\n" + "=" * 40)
        print("✓ ALL TESTS PASSED!")
        print("✓ TimeSeriesTransformer is working correctly")
        print("=" * 40)
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        raise


if __name__ == "__main__":
    main()
