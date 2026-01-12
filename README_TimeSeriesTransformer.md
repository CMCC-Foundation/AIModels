# TimeSeriesTransformer: A Comprehensive Transformer Model for Time Series Prediction

## Overview

The `TimeSeriesTransformer` is a PyTorch-based transformer model specifically designed for time series forecasting. It provides a complete solution for predicting future values in multivariate time series data using the power of transformer architecture.

## Features

- **Custom Transformer Architecture**: Optimized for time series data with positional encoding and multi-head attention
- **Multiple Loss Functions**: MSE, MAE, Huber, and custom loss functions
- **Complete Training Pipeline**: Training, validation, and testing with early stopping
- **Greedy Inference**: Autoregressive prediction for long sequences
- **Flexible Configuration**: Configurable model parameters for different use cases
- **Data Preprocessing**: Built-in data normalization and augmentation
- **Visualization Tools**: Training history and prediction plots
- **Real-world Ready**: Examples with synthetic and real-world data

## Architecture

The model consists of:

1. **Input Projection**: Linear layer to project input features to model dimension
2. **Positional Encoding**: Sinusoidal positional encoding for sequence order
3. **Transformer Layers**: Multiple transformer blocks with self-attention
4. **Output Projection**: Linear layer to project to output features

### Key Components

- **MultiHeadAttention**: Scaled dot-product attention with multiple heads
- **TransformerBlock**: Self-attention + feed-forward with residual connections
- **PositionalEncoding**: Sinusoidal encoding for sequence positions

## Installation

```bash
# Required dependencies
pip install torch numpy matplotlib scikit-learn pandas
```

## Quick Start

### Basic Usage

```python
import torch
from TimeSeriesTransformer import TimeSeriesTransformer, TimeSeriesTrainer, TimeSeriesPredictor
from sklearn.preprocessing import StandardScaler
import numpy as np

# Generate sample data
data = np.random.randn(1000, 5)  # 1000 time steps, 5 features
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data)

# Model parameters
TIN = 50  # Input sequence length
T = 10    # Output sequence length
M = 5     # Input features
N = 5     # Output features

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

# Create trainer and train
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trainer = TimeSeriesTrainer(model, device)

# Train model (you'll need to create data loaders first)
train_losses, val_losses = trainer.train(
    train_loader, val_loader,
    epochs=50,
    lr=0.001,
    loss_type='mse'
)

# Make predictions
predictor = TimeSeriesPredictor(model, device)
predictions = predictor.predict(input_sequence, T_predict=20)
```

### Complete Example

```python
from TimeSeriesTransformer import (
    TimeSeriesTransformer, 
    TimeSeriesDataset, 
    TimeSeriesTrainer, 
    TimeSeriesPredictor,
    create_data_loaders
)
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np

# 1. Prepare data
data = np.random.randn(1000, 5)
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data)

# 2. Create data loaders
TIN, T = 50, 10
train_loader, val_loader, test_loader = create_data_loaders(
    data_normalized, TIN, T, batch_size=32
)

# 3. Create model
model = TimeSeriesTransformer(
    input_size=5,
    output_size=5,
    d_model=128,
    n_heads=8,
    n_layers=4,
    d_ff=512,
    dropout=0.1
)

# 4. Train model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trainer = TimeSeriesTrainer(model, device)
train_losses, val_losses = trainer.train(
    train_loader, val_loader,
    epochs=50,
    lr=0.001,
    patience=10,
    loss_type='mse'
)

# 5. Evaluate model
predictor = TimeSeriesPredictor(model, device)
results, predictions, targets = predictor.evaluate(test_loader)
print(f"Test MSE: {results['mse']:.6f}")

# 6. Make predictions
sample_input = torch.FloatTensor(data_normalized[:TIN]).unsqueeze(0)
predictions = predictor.greedy_inference(sample_input, T_predict=20)
```

## Model Parameters

### TimeSeriesTransformer

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_size` | int | - | Number of input features (M) |
| `output_size` | int | - | Number of output features (N) |
| `d_model` | int | 128 | Model dimension |
| `n_heads` | int | 8 | Number of attention heads |
| `n_layers` | int | 6 | Number of transformer layers |
| `d_ff` | int | 512 | Feed-forward dimension |
| `dropout` | float | 0.1 | Dropout rate |
| `max_seq_len` | int | 1000 | Maximum sequence length |

### Training Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `epochs` | int | 100 | Number of training epochs |
| `lr` | float | 0.001 | Learning rate |
| `weight_decay` | float | 1e-5 | Weight decay |
| `patience` | int | 10 | Early stopping patience |
| `clip_grad` | float | 1.0 | Gradient clipping value |
| `loss_type` | str | 'mse' | Loss function type |
| `alpha` | float | 0.5 | Alpha for custom loss |

## Loss Functions

The model supports multiple loss functions:

1. **MSE Loss**: Mean Squared Error (default)
2. **MAE Loss**: Mean Absolute Error
3. **Huber Loss**: Huber loss with configurable delta
4. **Custom Loss**: Combination of MSE and MAE with alpha parameter

```python
# Example with different loss functions
trainer.train(train_loader, val_loader, loss_type='mae')
trainer.train(train_loader, val_loader, loss_type='huber')
trainer.train(train_loader, val_loader, loss_type='custom', alpha=0.7)
```

## Greedy Inference

For autoregressive prediction of long sequences, use greedy inference:

```python
# Predict 50 time steps using greedy inference
predictions = predictor.greedy_inference(input_sequence, T_predict=50)
```

The greedy inference works by:
1. Making a prediction for the next time step
2. Adding the prediction to the input sequence
3. Shifting the sequence and repeating
4. Continuing until the desired length is reached

## Data Format

### Input Data
- Shape: `(L, M)` where L is the number of time steps and M is the number of features
- Type: numpy array or torch tensor
- Preprocessing: Should be normalized (e.g., using StandardScaler)

### Model Input
- Shape: `(batch_size, TIN, M)` where TIN is the input sequence length
- Type: torch tensor

### Model Output
- Shape: `(batch_size, T, N)` where T is the output sequence length and N is the number of output features
- Type: torch tensor

## Examples

### 1. Basic Usage
Run the basic example with synthetic data:
```python
python transformer_example.py
```

### 2. Different Loss Functions
Compare different loss functions:
```python
from transformer_example import example_different_losses
example_different_losses()
```

### 3. Model Configurations
Test different model sizes:
```python
from transformer_example import example_model_configurations
example_model_configurations()
```

### 4. Greedy Inference
Demonstrate greedy inference:
```python
from transformer_example import example_greedy_inference
example_greedy_inference()
```

### 5. Real-world Data
Use with real-world datasets:
```python
from transformer_example import example_real_world_data
example_real_world_data()
```

## Advanced Usage

### Custom Dataset

```python
class CustomTimeSeriesDataset(TimeSeriesDataset):
    def __init__(self, data, TIN, T, additional_features=None):
        super().__init__(data, TIN, T)
        self.additional_features = additional_features
    
    def __getitem__(self, idx):
        input_seq, target_seq = super().__getitem__(idx)
        # Add custom processing here
        return input_seq, target_seq
```

### Custom Loss Function

```python
def custom_loss_function(pred, target, weight=1.0):
    mse = torch.mean((pred - target) ** 2)
    mae = torch.mean(torch.abs(pred - target))
    return weight * mse + (1 - weight) * mae

# Use in training
trainer.train(train_loader, val_loader, loss_type='custom')
```

### Model Saving and Loading

```python
# Save model
torch.save(model.state_dict(), 'model.pth')

# Load model
model = TimeSeriesTransformer(input_size=5, output_size=5)
model.load_state_dict(torch.load('model.pth'))
```

## Performance Tips

1. **Data Normalization**: Always normalize your data before training
2. **Batch Size**: Use larger batch sizes if memory allows
3. **Learning Rate**: Start with 0.001 and adjust based on convergence
4. **Early Stopping**: Use patience to prevent overfitting
5. **Gradient Clipping**: Helps with training stability
6. **Model Size**: Start with smaller models and scale up as needed

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or model size
2. **Slow Training**: Use GPU if available, reduce model complexity
3. **Poor Performance**: Check data normalization, try different loss functions
4. **Overfitting**: Increase dropout, reduce model size, use early stopping

### Debugging

```python
# Check model parameters
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Check data shapes
for batch in train_loader:
    input_seq, target_seq = batch
    print(f"Input shape: {input_seq.shape}")
    print(f"Target shape: {target_seq.shape}")
    break

# Check gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm = {param.grad.norm()}")
```

## Contributing

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{timeseriestransformer,
  title={TimeSeriesTransformer: A Comprehensive Transformer Model for Time Series Prediction},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/timeseriestransformer}
}
```

## Acknowledgments

- PyTorch team for the excellent deep learning framework
- The transformer architecture from "Attention Is All You Need"
- The time series community for inspiration and feedback
