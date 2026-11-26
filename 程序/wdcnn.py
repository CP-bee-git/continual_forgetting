"""WDCNN baseline for 2048-point, single-channel vibration windows."""

from torch import Tensor, nn


class WDCNN(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.ZeroPad1d((24, 24)),
            nn.Conv1d(1, 16, kernel_size=64, stride=16),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 3, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Linear(100, num_classes),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim == 2:
            inputs = inputs.unsqueeze(1)
        if inputs.ndim != 3 or inputs.shape[1:] != (1, 2048):
            raise ValueError("WDCNN expects input shaped (batch, 2048) or (batch, 1, 2048)")
        return self.classifier(self.features(inputs))


def _self_check() -> None:
    import torch

    model = WDCNN(num_classes=10)
    assert model(torch.zeros(2, 2048)).shape == (2, 10)
    assert model(torch.zeros(2, 1, 2048)).shape == (2, 10)


if __name__ == "__main__":
    _self_check()
