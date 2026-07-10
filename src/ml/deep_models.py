"""
Small sequence models for the Fast Lab fair trial (Part C).

Deliberately small (<200k parameters): at ~50k training rows per window,
big architectures would only memorize faster. Both models take
(batch, SEQ_LEN, n_channels) and emit 3-class logits. Training uses class
weights, early stopping on the validation slice, and Apple-Silicon MPS
when available.
"""

import logging
from typing import Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH = True
    DEVICE = ('mps' if torch.backends.mps.is_available()
              else 'cuda' if torch.cuda.is_available() else 'cpu')
except ImportError:
    TORCH = False
    DEVICE = 'none'


if TORCH:
    class SmallLSTM(nn.Module):
        def __init__(self, n_channels: int, hidden: int = 48,
                     n_classes: int = 3):
            super().__init__()
            self.lstm = nn.LSTM(n_channels, hidden, num_layers=1,
                                batch_first=True)
            self.head = nn.Sequential(
                nn.LayerNorm(hidden), nn.Dropout(0.2),
                nn.Linear(hidden, n_classes))

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1])

    class SmallCNN(nn.Module):
        def __init__(self, n_channels: int, n_classes: int = 3):
            super().__init__()
            # pooling arithmetic kept divisible for MPS: 60 -> 30 -> 5
            self.net = nn.Sequential(
                nn.Conv1d(n_channels, 32, kernel_size=5, padding=2),
                nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(32, 64, kernel_size=5, padding=2),
                nn.ReLU(), nn.AvgPool1d(kernel_size=6),
                nn.Flatten(), nn.Dropout(0.2),
                nn.Linear(64 * 5, n_classes))

        def forward(self, x):                     # (b, seq, ch)
            return self.net(x.transpose(1, 2))    # conv wants (b, ch, seq)


def train_deep(model_name: str, X_train, y_train, X_val, y_val,
               max_epochs: int = 8, batch_size: int = 512,
               lr: float = 1e-3, patience: int = 2,
               seed: int = 42) -> Tuple[object, Dict]:
    """Train one sequence model with early stopping on validation macro-F1.
    Returns (model in eval mode, info dict)."""
    if not TORCH:
        raise RuntimeError("torch not available")
    from sklearn.metrics import f1_score

    torch.manual_seed(seed)
    np.random.seed(seed)

    n_channels = X_train.shape[2]
    model = (SmallLSTM(n_channels) if model_name == 'lstm'
             else SmallCNN(n_channels)).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())

    counts = np.bincount(y_train, minlength=3).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1) / 3
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, device=DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_tr = torch.tensor(X_train, device=DEVICE)
    y_tr = torch.tensor(y_train, dtype=torch.long, device=DEVICE)
    X_va = torch.tensor(X_val, device=DEVICE)

    best_f1, best_state, bad_epochs = -1.0, None, 0
    n = len(X_tr)
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            optimizer.zero_grad()
            loss = criterion(model(X_tr[idx]), y_tr[idx])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_va).argmax(dim=1).cpu().numpy()
        f1 = f1_score(y_val, val_pred, average='macro')
        if f1 > best_f1 + 1e-4:
            best_f1, bad_epochs = f1, 0
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model, {'n_params': n_params, 'val_f1': best_f1,
                   'epochs_run': epoch + 1, 'device': DEVICE}


def predict_proba_deep(model, X, batch_size: int = 4096) -> np.ndarray:
    with torch.no_grad():
        out = []
        for start in range(0, len(X), batch_size):
            xb = torch.tensor(X[start:start + batch_size], device=DEVICE)
            out.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    return np.concatenate(out)
