import torch
import torch.nn as nn

X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

X_test = torch.tensor([5], dtype=torch.float32)
n_samples, n_features = X.shape

#model = nn.Linear(n_features, n_features) #input size, output size

class LinearRegression(nn.Module):
    def __init__(self, input_dimensions, output_dimensions):
        super(LinearRegression, self).__init__()
        #define layers
        self.lin = nn.Linear(input_dimensions, output_dimensions)
    
    def forward(self, x):
        return self.lin(x)

model = LinearRegression(n_features, n_features)

print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')

#training
learning_rate = 0.01
n_iters = 100

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(n_iters):
    y_pred = model(X) # forward pass
    l = loss(Y, y_pred) # loss
    l.backward() # gradients
    optimizer.step() #update weights
    optimizer.zero_grad() #zero gradients

    if epoch % 10 == 0:
        [w, b] = model.parameters()
        print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')

'''
    torch.randn()
    .numpy()
    to.()
    torch.from_numpy()
    torch.device()
    .backward()
    .grad.zero_()
'''