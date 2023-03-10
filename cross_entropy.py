import torch
import torch.nn as nn
import numpy as np

loss = nn.CrossEntropyLoss() # BCE for binary

Y = torch.tensor([0]) # not one-hot encoded
# n_samples x n_classes = 1x3
Y_pred_good = torch.tensor([[2.0, 1.0, 0.1]])
Y_pred_bad = torch.tensor([[0.5, 2.0, 0.3]])

l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)

print(l1.item())
print(l2.item())

_, prediction1 = torch.max(Y_pred_good, 1)
_, prediction2 = torch.max(Y_pred_bad, 1)

print(prediction1)
print(prediction2)