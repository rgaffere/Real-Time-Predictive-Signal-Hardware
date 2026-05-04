import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


T = 512
IN_CHANNELS = 6
OUT_CHANNELS = 6
CPL = 16
LAYER_COUNT = 7
K = 3
DILATIONS = [1, 2, 4, 8, 16, 32, 64]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
        return y[:, :, -1]


def load_imu_csv(path):
    df = pd.read_csv(path, header=None)

    # acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
    return df.iloc[:, [10, 11, 12, 4, 5, 6]].values.astype(np.float32)


def load_json_weights(model, weights_path):
    with open(weights_path, "r") as f:
        w = json.load(f)

    model.input_layer.conv1.conv.weight.data = torch.tensor(w["inputFilter1"], dtype=torch.float32)
    model.input_layer.conv1.conv.bias.data = torch.tensor(w["inputBias1"], dtype=torch.float32)

    model.input_layer.conv2.conv.weight.data = torch.tensor(w["inputFilter2"], dtype=torch.float32)
    model.input_layer.conv2.conv.bias.data = torch.tensor(w["inputBias2"], dtype=torch.float32)

    model.input_layer.residual.weight.data = torch.tensor(w["inputResidualFilter"], dtype=torch.float32).unsqueeze(-1)
    model.input_layer.residual.bias.data = torch.tensor(w["inputResidualBias"], dtype=torch.float32)

    for i, layer in enumerate(model.hidden_layers):
        layer.conv1.conv.weight.data = torch.tensor(w["hiddenFilter1"][i], dtype=torch.float32)
        layer.conv1.conv.bias.data = torch.tensor(w["hiddenBias1"][i], dtype=torch.float32)

        layer.conv2.conv.weight.data = torch.tensor(w["hiddenFilter2"][i], dtype=torch.float32)
        layer.conv2.conv.bias.data = torch.tensor(w["hiddenBias2"][i], dtype=torch.float32)

    model.output_layer.weight.data = torch.tensor(w["outputFilter"], dtype=torch.float32).unsqueeze(-1)
    model.output_layer.bias.data = torch.tensor(w["outputBiases"], dtype=torch.float32)

    mean = np.array(w["mean"], dtype=np.float32)
    std = np.array(w["std"], dtype=np.float32)

    return mean, std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out", default="pytorch_reference.csv")
    args = parser.parse_args()

    model = StreamingTCN().to(DEVICE)
    mean, std = load_json_weights(model, args.weights)
    model.eval()

    data = load_imu_csv(args.csv)
    data = (data - mean) / std

    with open(args.out, "w") as f:
        f.write("sample,pred0,pred1,pred2,pred3,pred4,pred5\n")

        with torch.no_grad():
            for sample in range(T, len(data)):
                x = data[sample - T + 1: sample + 1]
                x = torch.tensor(x.T, dtype=torch.float32).unsqueeze(0).to(DEVICE)

                pred = model(x).cpu().numpy()[0]

                f.write(str(sample))
                for ch in range(OUT_CHANNELS):
                    f.write(f",{pred[ch]:.10f}")
                f.write("\n")

    print(f"Saved PyTorch predictions to {args.out}")


if __name__ == "__main__":
    main()