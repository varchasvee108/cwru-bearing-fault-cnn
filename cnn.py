"""1D CNN classifier for vibration windows."""

from torch import nn


def conv_block(c_in, c_out, kernel, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv1d(c_in, c_out, kernel, stride=stride, padding=padding, bias=False),
        nn.BatchNorm1d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool1d(2),
    )


class CNN1D(nn.Module):
    def __init__(self, num_classes=10, widths=(32, 64, 128), first_kernel=64, first_stride=8, dropout=0.3):
        super().__init__()
        layers = [conv_block(1, widths[0], first_kernel, first_stride, padding=(first_kernel - first_stride) // 2)]
        layers += [conv_block(a, b, 3, padding=1) for a, b in zip(widths[:-1], widths[1:])]
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(widths[-1], num_classes))

    def forward(self, x):
        return self.head(self.pool(self.features(x)).flatten(1))
