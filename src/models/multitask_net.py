"""Multitask neural network: one shared trunk, three output heads.

    RSSI features
         |
    Linear -> LayerNorm -> GELU -> Dropout      (512)
    Linear -> LayerNorm -> GELU -> Dropout      (256)
    Linear -> LayerNorm -> GELU                 (128)   = shared representation
         |
    +----------+-----------+
    |          |           |
 building    floor    coordinates
  logits     logits      (x, y)
"""

import torch.nn as nn

ALL_TASKS = ("building", "floor", "coords")


def block(n_in, n_out, dropout):
    """Linear -> LayerNorm -> GELU (-> Dropout)."""
    layers = [nn.Linear(n_in, n_out), nn.LayerNorm(n_out), nn.GELU()]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class MultitaskNet(nn.Module):
    """Shared trunk with one head per task.

    tasks: which heads to build. ("building", "floor", "coords") is the multitask model;
           a single task, e.g. ("floor",), gives the single-task baseline with the same trunk.
    """

    def __init__(self, n_inputs, n_buildings=3, n_floors=5, hidden=(512, 256, 128),
                 dropout=0.2, tasks=ALL_TASKS):
        super().__init__()
        self.tasks = tuple(tasks)

        self.trunk = nn.Sequential(
            block(n_inputs, hidden[0], dropout),
            block(hidden[0], hidden[1], dropout),
            block(hidden[1], hidden[2], 0),
        )

        self.heads = nn.ModuleDict()
        if "building" in self.tasks:
            self.heads["building"] = nn.Linear(hidden[2], n_buildings)
        if "floor" in self.tasks:
            self.heads["floor"] = nn.Linear(hidden[2], n_floors)
        if "coords" in self.tasks:
            self.heads["coords"] = nn.Linear(hidden[2], 2)

    def forward(self, x):
        shared = self.trunk(x)
        # raw logits for classification (CrossEntropyLoss applies softmax itself)
        return {task: head(shared) for task, head in self.heads.items()}
