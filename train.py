import json
import math
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


# =========================
# Model config
# =========================

T = 512
IN_CHANNELS = 6
OUT_CHANNELS = 6
CPL = 16
LAYER_COUNT = 7
K = 3
DILATIONS = [1, 2, 4, 8, 16, 32, 64]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def receptive_field():
    return 1 + 2 * (K - 1) * sum(DILATIONS)


# =========================
# Dataset
# =========================

class MultiFileIMUDataset(torch.utils.data.Dataset):
    def __init__(self, arrays):
        self.arrays = [a.astype(np.float32) for a in arrays]
        self.index = []

        for file_idx, data in enumerate(self.arrays):
            n = len(data) - T
            for start_idx in range(n):
                self.index.append((file_idx, start_idx))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        file_idx, start_idx = self.index[idx]
        data = self.arrays[file_idx]

        x = data[start_idx : start_idx + T]
        y = data[start_idx + T]

        return torch.tensor(x.T), torch.tensor(y)


class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.pad = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size,
            dilation=dilation,
            padding=0
        )

    def forward(self, x):
        x = nn.functional.pad(x, (self.pad, 0))
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilation, use_projection):
        super().__init__()

        self.conv1 = CausalConv1d(in_ch, out_ch, K, dilation)
        self.conv2 = CausalConv1d(out_ch, out_ch, K, dilation)

        if use_projection:
            self.residual = nn.Conv1d(in_ch, out_ch, kernel_size=1)
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        res = self.residual(x)

        out = self.conv1(x)
        out = torch.relu(out)

        out = self.conv2(out)
        out = torch.relu(out)

        out = out + res

        return out


class StreamingTCN(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_layer = ResidualBlock(
            in_ch=IN_CHANNELS,
            out_ch=CPL,
            dilation=DILATIONS[0],
            use_projection=True
        )

        self.hidden_layers = nn.ModuleList([
            ResidualBlock(
                in_ch=CPL,
                out_ch=CPL,
                dilation=DILATIONS[i],
                use_projection=False
            )
            for i in range(1, LAYER_COUNT)
        ])

        self.output_layer = nn.Conv1d(CPL, OUT_CHANNELS, kernel_size=1)

    def forward(self, x):
        x = self.input_layer(x)

        for layer in self.hidden_layers:
            x = layer(x)

        y = self.output_layer(x)

        # latest point only, matching doOutputLayer()
        return y[:, :, -1]


# =========================
# Export
# =========================

def export_for_cpp(model, mean, std, path):
    m = model

    export = {
        "T": T,
        "inChannels": IN_CHANNELS,
        "layerCount": LAYER_COUNT,
        "cpl": CPL,
        "k": K,
        "dilations": DILATIONS,
        "outChannels": OUT_CHANNELS,

        "mean": mean.tolist(),
        "std": std.tolist(),

        "inputFilter1": m.input_layer.conv1.conv.weight.detach().cpu().numpy().tolist(),
        "inputBias1": m.input_layer.conv1.conv.bias.detach().cpu().numpy().tolist(),

        "inputFilter2": m.input_layer.conv2.conv.weight.detach().cpu().numpy().tolist(),
        "inputBias2": m.input_layer.conv2.conv.bias.detach().cpu().numpy().tolist(),

        "inputResidualFilter": m.input_layer.residual.weight.detach().cpu().numpy()[:, :, 0].tolist(),
        "inputResidualBias": m.input_layer.residual.bias.detach().cpu().numpy().tolist(),

        "hiddenFilter1": [],
        "hiddenBias1": [],
        "hiddenFilter2": [],
        "hiddenBias2": [],

        "outputFilter": m.output_layer.weight.detach().cpu().numpy()[:, :, 0].tolist(),
        "outputBiases": m.output_layer.bias.detach().cpu().numpy().tolist()
    }

    for layer in m.hidden_layers:
        export["hiddenFilter1"].append(
            layer.conv1.conv.weight.detach().cpu().numpy().tolist()
        )
        export["hiddenBias1"].append(
            layer.conv1.conv.bias.detach().cpu().numpy().tolist()
        )
        export["hiddenFilter2"].append(
            layer.conv2.conv.weight.detach().cpu().numpy().tolist()
        )
        export["hiddenBias2"].append(
            layer.conv2.conv.bias.detach().cpu().numpy().tolist()
        )

    with open(path, "w") as f:
        json.dump(export, f, indent=2)

# =========================
# Training
# =========================

def load_imu_csv(path):
    df = pd.read_csv(path, header=None)

    # directly select by column index
    data = df.iloc[:, [10, 11, 12, 4, 5, 6]].values.astype(np.float32)

    return data


def train(args):

    arrays = [load_imu_csv(path) for path in args.csv]

    all_data = np.concatenate(arrays, axis=0)
    mean = all_data.mean(axis=0)
    std = all_data.std(axis=0) + 1e-8

    arrays = [(a - mean) / std for a in arrays]

    rf = receptive_field()
    print(f"Receptive field: {rf} samples")

    split = int(len(arrays) * 0.8)

    train_arrays = arrays[:split]
    val_arrays = arrays[split:]

    train_set = MultiFileIMUDataset(train_arrays)
    val_set = MultiFileIMUDataset(val_arrays)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    model = StreamingTCN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best_val = math.inf

    patience = 8
    bad_epochs = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            pred = model(x)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(DEVICE)
                y = y.to(DEVICE)

                pred = model(x)
                loss = loss_fn(pred, y)

                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1:03d} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            bad_epochs = 0
            torch.save(model.state_dict(), args.model_out)
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f"Best validation MSE: {best_val:.6f}")

    model.load_state_dict(torch.load(args.model_out, map_location=DEVICE))
    export_for_cpp(model, mean, std, args.export_json)

    plot_validation(model, val_set, mean, std, args.plot_out)


# =========================
# Plot validation
# =========================

def plot_validation(model, val_set, mean, std, plot_path):
    model.eval()

    preds = []
    actuals = []

    with torch.no_grad():
        for i in range(min(1000, len(val_set))):
            x, y = val_set[i]
            x = x.unsqueeze(0).to(DEVICE)

            pred = model(x).cpu().numpy()[0]

            preds.append(pred)
            actuals.append(y.numpy())

    preds = np.array(preds)
    actuals = np.array(actuals)

    preds_denorm = preds * std + mean
    actuals_denorm = actuals * std + mean

    labels = [
        "user_acc_x(G)", "user_acc_y(G)", "user_acc_z(G)",
        "rotation_rate_x(radians_s)", "rotation_rate_y(radians_s)", "rotation_rate_z(radians_s)"
    ]

    for ch in range(6):
        plt.figure()
        plt.plot(actuals_denorm[:, ch], label="Actual")
        plt.plot(preds_denorm[:, ch], label="Predicted")
        plt.title(f"TCN Prediction Validation: {labels[ch]}")
        plt.xlabel("Sample")
        plt.ylabel(labels[ch])
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{plot_path}_{labels[ch]}.png", dpi=200)
        plt.close()

    error = np.mean((preds_denorm - actuals_denorm) ** 2, axis=1)

    plt.figure()
    plt.plot(error)
    plt.title("Prediction Error Over Time")
    plt.xlabel("Sample")
    plt.ylabel("MSE")
    plt.grid(True)
    plt.savefig(f"{plot_path}_error.png", dpi=200)
    plt.close()

    print(f"Saved validation plots as {plot_path}_*.png")


# =========================
# Main
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", required=True, nargs="+")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--model_out", default="tcn_imu_model.pt")
    parser.add_argument("--export_json", default="tcn_imu_weights.json")
    parser.add_argument("--plot_out", default="validation")

    args = parser.parse_args()

    train(args)