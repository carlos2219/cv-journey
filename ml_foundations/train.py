import torch
from torch import nn
from torch.utils.data import DataLoader

from ml_foundations.crops_dataset import CropsDataset, NUM_CLASSES
from ml_foundations.model import SmallCNN

torch.manual_seed(0)
train_loader = DataLoader(CropsDataset("train"), batch_size=64, shuffle=True)
val_loader = DataLoader(CropsDataset("val"), batch_size=256)

model = SmallCNN(NUM_CLASSES)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(5):
    model.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        Y_hat = model(xb)
        L = loss_fn(Y_hat, yb)
        running_loss += L.item() * len(yb)
        L.backward()
        optimizer.step()
    model.eval()
    correct = 0
    cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64)  # rows = true, cols = predicted
    total = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            logits = model(xb)
            preds = logits.argmax(dim=1)
            for t, p in zip(yb, preds):
                cm[t, p] += 1
            correct += (preds == yb).sum().item()
            total += yb.numel()
    print(cm)
    print("recall per class:", cm.diag() / cm.sum(dim=1))
    print(f"epoch {epoch}  train loss {running_loss / len(train_loader.dataset):.3f}  val acc {correct / total:.3f}")
