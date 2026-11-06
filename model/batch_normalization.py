import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists


        # the first thing we do is to try to get the mean by looping whole batch
        x = np.array(x)
        gamma = np.array(gamma)
        beta = np.array(beta)
        running_mean = np.array(running_mean)
        running_var = np.array(running_var)
        if training:
            u=[]
            featurenum=len(x[0])
            batchsize=len(x)
            for i in range(featurenum):
                sum=0
                for j in range(batchsize):
                    sum+=x[j][i]
                u.append(1/batchsize*sum)
                    
            var=[]
            var=1/batchsize*np.sum((np.array(x) - np.array(u))**2,axis=0)
            x_hat= (np.array(x) - np.array(u))/np.sqrt(var+eps)
            # y=gamma*x_hat+beta
            running_mean=(1-momentum)*np.array(running_mean)+momentum*np.array(u)
            running_var=(1-momentum)*np.array(running_var)+momentum*np.array(var)
        else:
            x_hat = (x - running_mean) / np.sqrt(running_var + eps)
        
        y = gamma * x_hat + beta

        return (
            np.round(y, 4),
            np.round(running_mean, 4),
            np.round(running_var, 4)
        )