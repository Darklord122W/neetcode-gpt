import torch
import torch.nn as nn
from typing import List


class Solution:

    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        # Forward pass through the model.
        # After each ReLU layer, compute the fraction of neurons that are dead.
        # A neuron is dead if it outputs 0 for ALL samples in the batch.
        # Return a list of dead fractions (one per ReLU layer), rounded to 4 decimals.
        stats = []

        for layer in model:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                
                # mean = x.mean().item()
                # std = x.std().item()
                dead_fraction = (x <= 0).all(dim=0).float().mean().item()
                # stats.append({
                #     # "mean": round(mean, 4),
                #     # "std": round(std, 4),
                #     "dead_fraction": round(dead_fraction, 4)
                # })
                stats.append(round(dead_fraction, 4))
        return stats        

    def suggest_fix(self, dead_fractions: List[float]) -> str:
        # Given dead fractions per ReLU layer, suggest a fix.
        # Check in this order:
        # 1. 'use_leaky_relu' if any layer has dead fraction > 0.5
        # 2. 'reinitialize' if the first layer has dead fraction > 0.3
        # 3. 'reduce_learning_rate' if dead fraction strictly increases
        #    with depth AND the last layer's fraction > 0.1
        # 4. 'healthy' if max dead fraction < 0.1
        # 5. 'healthy' otherwise
        checks=[]
        for i, layer in enumerate(dead_fractions):
            
            if layer>0.5:
                return "use_leaky_relu"
            if i==0 and layer>0.3:
                return "reinitialize"
        for i in range(1,len(dead_fractions)):
            checks.append(dead_fractions[i]>dead_fractions[i-1])
        if all(checks) and dead_fractions[-1]>0.1:
            return "reduce_learning_rate"
        if max(dead_fractions)<0.1:
            return "healthy"
        return "healthy"
            
        # for layer in gradient_stats:
        #     if layer['norm']>1000:
        #         return "exploding_gradients"
        #     if layer['norm']<1e-5:
        #         return "vanishing_gradients"   
        # for layer in activation_stats:
        #     if layer['std'] < 0.1:
        #         return 'vanishing_gradients'                            
        #     if layer['std'] > 10.0:
        #         return 'exploding_gradients'

            

        return "healthy"       
