import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

a=torch.tensor(2.)
b=torch.tensor(6.)
c=(a-b).mean()

print(c)
d=torch.tensor([[1],[2],[3]])
print(d.size())

e=torch.tensor([[1,2],[3,4]])
print(e.size())


# y=torch.tensor([1,9,3,4])
# print(y.shape)
# y=y.view(y.shape[0],1)
# print(y)



# 0) Prepare data
X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)

# cast to float Tensor
X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0], 1)

print(X)