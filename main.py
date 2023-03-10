import torch
import numpy as np

X = np.array([1, 2, 3, 4], dtype=np.float32)
Y = np.array([2, 4, 6, 8], dtype=np.float32)
w = 0.0

#model prediction
def forward(x):
    return w*x

#loss(MSE)
def loss(y, y_predicted):
    return ((y_predicted - y)**2).mean()

#gradient
def gradient(x, y, y_predicted):
    return np.dot(2 * x, y_predicted - y).mean()

print(f'Prediction before training: f(5) = {forward(5):.3f}')

#training
learning_rate = 0.01
n_iters = 20

for epoch in range(n_iters):
    y_pred = forward(X) # forward pass
    l = loss(Y, y_pred) # loss
    dw = gradient(X, Y, y_pred) # gradients
    w -= learning_rate * dw # update weights
    if epoch % 2 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {forward(5):.3f}')

'''
    torch.randn()
    .numpy()
    to.()
    torch.from_numpy()
    torch.device()
    .backward()
    .grad.zero_()
'''