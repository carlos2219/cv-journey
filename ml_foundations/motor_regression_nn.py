from torch import nn
import torch

torch.manual_seed(0)
N = 200 # bench runs

#                     speed   current
W_true = torch.tensor([[ 300.0, 0.2],     # voltage
                       [-150.0, 2.0],     # torque
                       [ -10.0, 0.1]])    # temp
b_true = torch.tensor([1500.0, 3.0])      # nominal speed (rpm), current (A)

X = torch.randn(N, 3)                     # standardized sensor readings
noise = torch.randn(N, 2) * torch.tensor([20.0, 0.1])   # sensor noise per output  ±20 rpm on speed and ±0.1 A on current
Y = X @ W_true + b_true + noise           # measured speed and current, (N, 2)

loss_fn = nn.MSELoss()
model = nn.Linear(in_features=3, out_features=2) # owns weight (2,3) and bias (2,)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1) # parameters(): all knobs of the model

for step in range(100):
    optimizer.zero_grad()
    Y_hat = model(X)
    L = loss_fn(Y_hat, Y)
    L.backward()
    optimizer.step()
    if step % 10 == 0:
        print(f"step {step:3d}  loss {L.item():12.2f}")

print("weight shape:", model.weight.shape)
print("W learned (transposed back):\n", model.weight.detach().T)
print("b learned:", model.bias.detach())