import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

from data_preprocess import prepare_data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LR = 0.1
EPOCHS = 8192

STRIDE = 6
OVERLAP = 3

# model
SEQ_LEN = STRIDE
POINT_NUM = 50
POINT_DIM = 10
HIDDEN_RATIO = 2.0
DROPOUT_RATIO = 0.0
NUM_LAYERS = 1


class MLP_MODULE(nn.Module):
    def __init__(self, input_dim, hidden_ratio=2.0, dropout_ratio=0.1):
        super(MLP_MODULE, self).__init__()
        hidden_dim = int(hidden_ratio * input_dim)
        self.up_proj = nn.Linear(input_dim, hidden_dim)
        self.down_proj = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, x):
        # x = nn.functional.layer_norm(x, x.shape[-1:])
        tmp = self.down_proj(nn.functional.gelu(self.up_proj(x)))
        tmp = self.dropout(tmp) + x
        return tmp


class MLP_LAYER(nn.Module):
    def __init__(self, seq_len=6, point_num=50, f_dim=10, hidden_ratio=2.0, dropout_ratio = 0.1):
        super(MLP_LAYER, self).__init__()
        self.mlp_module1 = MLP_MODULE(f_dim, hidden_ratio, dropout_ratio)
        self.mlp_module2 = MLP_MODULE(point_num, hidden_ratio, dropout_ratio)
        self.mlp_module3 = MLP_MODULE(seq_len, hidden_ratio, dropout_ratio)
    
    def forward(self, x):
        tmp = self.mlp_module1(x)
        tmp = tmp.permute(0, 3, 1, 2)
        tmp = self.mlp_module2(tmp)
        tmp = tmp.permute(0, 3, 1, 2)
        tmp = self.mlp_module3(tmp)
        tmp = tmp.permute(0, 3, 1, 2)
        return tmp
    

class EMBEDDING(nn.Module):
    def __init__(self, point_num=50, f_dim=10):
        super(EMBEDDING, self).__init__()
        self.time_num = 8
        self.point_num = point_num
        self.f_num = f_dim

        self.time_embedding = nn.Embedding(self.time_num, 1)
        self.point_embedding = nn.Embedding(self.point_num, 1)
        self.f_embedding = nn.Embedding(self.f_num, 1)

    def forward(self, x, time_list): # x (batch, seq_len, point_num, f_dim)
        time_e = self.time_embedding(
            (torch.tensor(time_list).reshape(
                [x.size(0), x.size(1), 1, 1]
            ).repeat(1, 1, x.size(2), x.size(3))/3).long().to(x.device)
        ).squeeze()
        point_e = self.point_embedding(
            torch.arange(self.point_num).reshape(
                [1, 1, -1, 1]
            ).repeat(x.size(0), x.size(1), 1, x.size(3)).long().to(x.device)
        ).squeeze()
        f_e = self.f_embedding(
            torch.arange(self.f_num).reshape(
                [1, 1, 1, -1]
            ).repeat(x.size(0), x.size(1), x.size(2), 1).long().to(x.device)
        ).squeeze()
        return x + time_e + point_e + f_e


class MLP_Net(nn.Module):
    def __init__(
            self,
            seq_len=6,
            point_num=50,
            point_dim=10,
            hidden_ratio=2.0,
            dropout_ratio=0.1,
            num_layers=2
    ):
        super(MLP_Net, self).__init__()
        self.embedding = EMBEDDING(point_num, point_dim)
        self.mlp_layers = nn.ModuleList(
            [
                MLP_LAYER(seq_len, point_num, point_dim, hidden_ratio, dropout_ratio)
                for _ in range(num_layers)
            ]
        )
        self.head = nn.Linear(
            seq_len * point_num * point_dim,
            seq_len * point_num * point_dim
        )

    def forward(self, x, time_list):
        tmp = self.embedding(x, time_list)
        for layer in self.mlp_layers:
            tmp = layer(tmp)
        tmp = self.head(tmp.reshape([x.size(0), -1])).reshape(x.size())
        return tmp


def data_collate_fn(batch):
    x = [
        torch.from_numpy(item["gathered_input"])
        for item in batch
    ]
    y = [
        torch.from_numpy(item["gathered_target_abs"])
        for item in batch
    ]
    time_list = [
        [
            int(_[-2:])
            for _ in item["dates"]
        ]
        for item in batch
    ]
    x = torch.stack(x).float()
    y = torch.stack(y).float()
    return {"x": x, "y": y, "time_list": time_list}


def get_dataloader():
    data = prepare_data(
        stride=STRIDE,
        overlap=OVERLAP,
    )

    print("=====data_keys=====")
    print(data["train_data"][0].keys())
    print("===================")

    train_loader = DataLoader(
        data["train_data"],
        batch_size=len(data["train_data"]),
        shuffle=True,
        collate_fn=data_collate_fn
    )
    eval_loader = DataLoader(
        data["eval_data"],
        batch_size=len(data["eval_data"]),
        shuffle=False,
        collate_fn=data_collate_fn,
    )
    return train_loader, eval_loader


def loss_figure(train_losses, eval_losses):
    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(eval_losses, label="eval")
    plt.legend()
    plt.savefig("loss.png")
    plt.close()


# Define the training function
def run():
    train_loader, eval_loader = get_dataloader()
    model = MLP_Net(
        seq_len=STRIDE,
        point_num=POINT_NUM,
        point_dim=POINT_DIM,
        hidden_ratio=HIDDEN_RATIO,
        dropout_ratio=DROPOUT_RATIO,
        num_layers=NUM_LAYERS
    ).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    train_losses = []
    eval_losses = []
    for epoch in range(EPOCHS):
        model.train()
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            x, y = batch["x"].to(DEVICE), batch["y"].to(DEVICE)
            y_pred = model(x, batch["time_list"])
            loss = criterion(
                y_pred.reshape([y_pred.size(0), -1]),
                y.reshape([y.size(0), -1])
            )
            train_losses.append(loss.item())
            tqdm.write(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}")
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            eval_loss = 0
            for batch in tqdm(eval_loader):
                x, y = batch["x"].to(DEVICE), batch["y"].to(DEVICE)
                y_pred = model(x, batch["time_list"])
                loss = criterion(
                    y_pred.reshape([y_pred.size(0), -1]),
                    y.reshape([y.size(0), -1])
                )
                eval_loss += loss.item()
            eval_loss /= len(eval_loader)
            eval_losses.append(eval_loss)
            print(f"Epoch {epoch+1}/{EPOCHS}, Eval Loss: {eval_loss:.4f}")
        
        loss_figure(train_losses, eval_losses)


if __name__ == "__main__":
    run()
