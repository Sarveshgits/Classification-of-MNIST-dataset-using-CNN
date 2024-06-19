import torch
import numpy as np

a=torch.tensor(9.,requires_grad=True)
b=torch.tensor(3.,requires_grad=True)
y=a**3 - b**2
y.backward()
print(a.grad)