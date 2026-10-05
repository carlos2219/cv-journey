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

W = torch.zeros(3, 2, requires_grad=True)
b = torch.zeros(2, requires_grad=True)
lr = 0.1
print("lr: ",lr)

for step in range(100):
    Y_hat = X @ W + b                           # 1. forward
    L = ((Y_hat - Y) ** 2).sum(dim=1).mean()
    L.backward()                                # 2. backward
    with torch.no_grad():
        W -= lr * W.grad                        # 3. update
        b -= lr * b.grad
        W.grad.zero_()                          # 4. zero
        b.grad.zero_()

    if step % 10 == 0:
        print(f"step {step:3d} loss {L.item():12.2f}")     # .item(): 1-element tensor -> Python float

print("W learned: \n", W.detach()) # .detach(): drop grad tracking for printing
print("b learned: \n", b.detach())
