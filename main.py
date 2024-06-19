import torch
import numpy as np

a=torch.ones(5)
b=a.numpy()

a.add_(1)
print(a)
print(b)


c=torch.rand(3)
print(c)

