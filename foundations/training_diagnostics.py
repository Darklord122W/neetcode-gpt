import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        stats = []

        with torch.no_grad():
            for layer in model:
                x = layer(x)
                if isinstance(layer, nn.Linear):
                    
                    mean = x.mean().item()
                    std = x.std().item()
                    dead_fraction = (x <= 0).all(dim=0).float().mean().item()
                    stats.append({
                        "mean": round(mean, 4),
                        "std": round(std, 4),
                        "dead_fraction": round(dead_fraction, 4)
                    })

        return stats
        

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        stats = []
        model.zero_grad()

        prediction=model(x)
        loss_fn=nn.MSELoss()

        loss=loss_fn(prediction,y) #loss = nn.MSELoss()(prediction, y)
        loss.backward()

        for layer in model:
            if isinstance(layer, nn.Linear):    
                grad=layer.weight.grad
                ##layer.weight.grad  # Weight gradients: ∂L/∂Wlayer.bias.grad    # Bias gradients:   ∂L/∂b
                mean = grad.mean().item()
                std = grad.std().item()
                norm = torch.norm(grad).item()
                stats.append({
                    "mean": round(mean, 4),
                    "std": round(std, 4),
                    "norm": round(norm, 4)
                })

        return stats 

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        for layer in activation_stats:
            
            if layer['dead_fraction']>0.5:
                return "dead_neurons"
        for layer in gradient_stats:
            if layer['norm']>1000:
                return "exploding_gradients"
            if layer['norm']<1e-5:
                return "vanishing_gradients"   
        for layer in activation_stats:
            if layer['std'] < 0.1:
                return 'vanishing_gradients'                            
            if layer['std'] > 10.0:
                return 'exploding_gradients'

            

        return "healthy"

        
