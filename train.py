import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from models.transformer import AF2BindTransformer


class AF2BindDataset(Dataset):

    def __init__(self, data_dir="data"):

        self.samples = []

        for file in os.listdir(data_dir):

            if file.endswith(".pkl"):

                path = os.path.join(data_dir, file)

                with open(path, "rb") as f:
                    sample = pickle.load(f)

                self.samples.append(sample)

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        sample = self.samples[idx]

        x = torch.tensor(sample["x"], dtype=torch.float32)
        y = torch.tensor(sample["y"], dtype=torch.float32)

        return x, y


dataset = AF2BindDataset()

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True
)

model = AF2BindTransformer()

criterion = nn.BCELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4
)

for epoch in range(5):

    print(f"Epoch {epoch+1}")

    total_loss = 0

    for x, y in loader:

        pred = model(x)

        pred = pred.squeeze(-1)

        loss = criterion(pred, y)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print("loss =", total_loss)

torch.save(
    model.state_dict(),
    "af2bind_transformer.pt"
)

print("training finished")
