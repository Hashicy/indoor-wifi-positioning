"""Train the multitask network and the single-task baselines (Member 1).

Run from the repository root:
    python -m src.train_multitask

What it does
------------
1. prepare_data(): clean, split (train / val / test), scale, encode.
2. Trains the multitask model (building + floor + coordinates) with early stopping on the val set.
3. Trains three single-task models (same trunk, one head each) the same way.
4. Evaluates all four on val and, once, on the test set; saves checkpoints, histories and predictions.
"""

import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn

from src.data_preprocessing import prepare_data
from src.dataset import make_loader
from src.models.multitask_net import ALL_TASKS, MultitaskNet

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "models"
METRICS_DIR = REPO_ROOT / "results" / "metrics"

CONFIG = {
    "hidden": (512, 256, 128),
    "dropout": 0.2,
    "learning_rate": 1e-3,
    "weight_decay": 0.0,
    "batch_size": 128,
    "max_epochs": 150,
    "patience": 20,          # stop if the val score has not improved for this many epochs
    "coord_loss": "mse",     # "mse" or "smoothl1" (on standardised coordinates)
    "seed": 42,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ------------------------------------------------------------------ losses

cross_entropy = nn.CrossEntropyLoss()


def compute_losses(outputs, building, floor, coords, coord_loss="mse"):
    """One loss per task the model has, plus their sum as 'total'."""
    losses = {}
    if "building" in outputs:
        losses["building"] = cross_entropy(outputs["building"], building)
    if "floor" in outputs:
        losses["floor"] = cross_entropy(outputs["floor"], floor)
    if "coords" in outputs:
        if coord_loss == "smoothl1":
            losses["coords"] = nn.functional.smooth_l1_loss(outputs["coords"], coords)
        else:
            losses["coords"] = nn.functional.mse_loss(outputs["coords"], coords)
    losses["total"] = sum(losses.values())     # L_total = L_building + L_floor + L_coords
    return losses


def run_epoch(model, loader, config, optimizer=None):
    """One pass over the data. Trains if an optimizer is given, otherwise only evaluates."""
    training = optimizer is not None
    model.train(training)
    sums, n = {}, 0

    for X, building, floor, coords in loader:
        X, building, floor, coords = X.to(DEVICE), building.to(DEVICE), floor.to(DEVICE), coords.to(DEVICE)
        with torch.set_grad_enabled(training):
            losses = compute_losses(model(X), building, floor, coords, config["coord_loss"])
            if training:
                optimizer.zero_grad()
                losses["total"].backward()
                optimizer.step()
        for name, value in losses.items():
            sums[name] = sums.get(name, 0.0) + value.item() * len(X)
        n += len(X)

    return {name: total / n for name, total in sums.items()}


# ------------------------------------------------------------------ predictions and metrics

@torch.no_grad()
def predict(model, d, prep):
    """Predictions in the original label space and in metres."""
    model.eval()
    outputs = model(torch.tensor(d["X"]).to(DEVICE))
    pred = {}
    if "building" in outputs:
        pred["building"] = prep["building_encoder"].inverse_transform(outputs["building"].argmax(1).cpu().numpy())
    if "floor" in outputs:
        pred["floor"] = prep["floor_encoder"].inverse_transform(outputs["floor"].argmax(1).cpu().numpy())
    if "coords" in outputs:
        pred["coords_m"] = prep["coord_scaler"].inverse_transform(outputs["coords"].cpu().numpy())
    return pred


def compute_metrics(pred, d, prep):
    """Accuracies and positioning error (metres) for whatever the model predicts."""
    true_building = prep["building_encoder"].inverse_transform(d["building"])
    true_floor = prep["floor_encoder"].inverse_transform(d["floor"])
    m = {}
    if "building" in pred:
        m["building_acc"] = np.mean(pred["building"] == true_building)
    if "floor" in pred:
        m["floor_acc"] = np.mean(pred["floor"] == true_floor)
        m["floor_abs_diff"] = np.mean(np.abs(pred["floor"] - true_floor))   # floors off, on average
    if "building" in pred and "floor" in pred:
        m["joint_acc"] = np.mean((pred["building"] == true_building) & (pred["floor"] == true_floor))
    if "coords_m" in pred:
        errors = np.sqrt(((pred["coords_m"] - d["coords_m"]) ** 2).sum(axis=1))
        m["mean_error_m"] = errors.mean()
        m["median_error_m"] = np.median(errors)
        m["p90_error_m"] = np.percentile(errors, 90)
    return m


def selection_score(m):
    """Validation score used to pick the best epoch (lower is better).

    EvAAL/IPIN 2015 competition metric for UJIIndoorLoc:
        error = distance + 50 * (building wrong) + 4 * (number of floors off)
    averaged over scans. Single-task models use only the parts they predict.
    """
    score = 0.0
    if "mean_error_m" in m:
        score += m["mean_error_m"]
    if "building_acc" in m:
        score += 50 * (1 - m["building_acc"])
    if "floor_abs_diff" in m:
        score += 4 * m["floor_abs_diff"]
    return score


# ------------------------------------------------------------------ training

def train_model(data, prep, tasks=ALL_TASKS, name="multitask", config=CONFIG, verbose=True):
    """Train one model with early stopping on the val selection score. Saves the best checkpoint.

    Why not the val loss? The classification losses keep rising on val (over-confidence on a few
    hard scans) while accuracy stays flat and the coordinate error is still improving, so the
    summed loss would stop training far too early. The selection score measures what we care about.
    """
    set_seed(config["seed"])
    model = MultitaskNet(
        n_inputs=data["train"]["X"].shape[1],
        n_buildings=len(prep["building_encoder"].classes_),
        n_floors=len(prep["floor_encoder"].classes_),
        hidden=config["hidden"], dropout=config["dropout"], tasks=tasks,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])

    train_loader = make_loader(data["train"], config["batch_size"], shuffle=True)
    val_loader = make_loader(data["val"], config["batch_size"])

    MODEL_DIR.mkdir(exist_ok=True)
    checkpoint_path = MODEL_DIR / f"{name}_best.pth"
    best_score, best_epoch, history = float("inf"), 0, []
    start = time.time()

    for epoch in range(1, config["max_epochs"] + 1):
        train_losses = run_epoch(model, train_loader, config, optimizer)
        val_losses = run_epoch(model, val_loader, config)
        val_metrics = compute_metrics(predict(model, data["val"], prep), data["val"], prep)
        score = selection_score(val_metrics)

        row = {"epoch": epoch}
        row.update({f"train_{k}_loss": v for k, v in train_losses.items()})
        row.update({f"val_{k}_loss": v for k, v in val_losses.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        row["val_score"] = score
        history.append(row)

        if score < best_score:                               # best model so far: save it
            best_score, best_epoch = score, epoch
            torch.save({"model_state": model.state_dict(), "tasks": tasks, "config": config,
                        "n_inputs": data["train"]["X"].shape[1], "epoch": epoch}, checkpoint_path)

        if verbose and (epoch == 1 or epoch % 10 == 0):
            print(f"[{name}] epoch {epoch:3d}  train loss {train_losses['total']:.4f}  "
                  f"val loss {val_losses['total']:.4f}  val score {score:.3f}  " +
                  "  ".join(f"{k} {v:.3f}" for k, v in val_metrics.items()))

        if epoch - best_epoch >= config["patience"]:
            break

    # reload the best weights
    model.load_state_dict(torch.load(checkpoint_path)["model_state"])
    history = pd.DataFrame(history)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    history.to_csv(METRICS_DIR / f"{name}_training_history.csv", index=False)
    if verbose:
        print(f"[{name}] best epoch {best_epoch} (val score {best_score:.3f}), "
              f"stopped at epoch {len(history)}, {time.time() - start:.0f}s")
    return model, history


def load_model(name="multitask"):
    """Rebuild a saved model from its checkpoint."""
    ckpt = torch.load(MODEL_DIR / f"{name}_best.pth")
    cfg = ckpt["config"]
    model = MultitaskNet(ckpt["n_inputs"], hidden=cfg["hidden"], dropout=cfg["dropout"], tasks=ckpt["tasks"])
    model.load_state_dict(ckpt["model_state"])
    return model.to(DEVICE)


# ------------------------------------------------------------------ main experiment

MODELS = {
    "multitask": ALL_TASKS,
    "single_building": ("building",),
    "single_floor": ("floor",),
    "single_coords": ("coords",),
}


def main():
    data, prep = prepare_data()
    print("Sizes:", prep["sizes"], " Input features:", len(prep["used_waps"]))

    rows, models = [], {}
    for name, tasks in MODELS.items():
        model, history = train_model(data, prep, tasks=tasks, name=name)
        models[name] = model
        for split in ["val", "test"]:
            m = compute_metrics(predict(model, data[split], prep), data[split], prep)
            rows.append({"model": name, "split": split,
                         "best_epoch": int(history["val_score"].idxmin()) + 1,
                         **m, "evaal_score": selection_score(m)})

    comparison = pd.DataFrame(rows)
    comparison.to_csv(METRICS_DIR / "dl_multitask_vs_single_task.csv", index=False)

    # test-set predictions of the multitask model (for Member 4's evaluation)
    pred = predict(models["multitask"], data["test"], prep)
    test = data["test"]
    pd.DataFrame({
        "BUILDINGID": prep["building_encoder"].inverse_transform(test["building"]),
        "FLOOR": prep["floor_encoder"].inverse_transform(test["floor"]),
        "LONGITUDE": test["coords_m"][:, 0], "LATITUDE": test["coords_m"][:, 1],
        "pred_building": pred["building"], "pred_floor": pred["floor"],
        "pred_longitude": pred["coords_m"][:, 0], "pred_latitude": pred["coords_m"][:, 1],
        "position_error_m": np.sqrt(((pred["coords_m"] - test["coords_m"]) ** 2).sum(axis=1)),
    }).to_csv(METRICS_DIR / "multitask_test_predictions.csv", index=False)

    run_info = {
        "config": {k: list(v) if isinstance(v, tuple) else v for k, v in CONFIG.items()},
        "split_sizes": prep["sizes"],
        "n_input_features": len(prep["used_waps"]),
        "split_rule": "train/val: GroupShuffleSplit by position, 15% val, seed 42; test: validationData.csv",
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "sklearn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__},
        "device": str(DEVICE),
    }
    with open(METRICS_DIR / "dl_run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)

    pd.set_option("display.width", 160)
    print(comparison.round(4).to_string(index=False))


def compare_over_seeds(seeds=(42, 7, 123)):
    """Train every model with several random seeds, so the comparison isn't down to luck.

    Saves results/metrics/dl_seed_comparison.csv (one row per model, seed and split).
    Checkpoints are written as <model>_seed<seed>_best.pth so the main ones are not overwritten.
    """
    data, prep = prepare_data(save=False)
    rows = []
    for seed in seeds:
        config = dict(CONFIG, seed=seed)
        for name, tasks in MODELS.items():
            model, history = train_model(data, prep, tasks=tasks, name=f"{name}_seed{seed}",
                                         config=config, verbose=False)
            for split in ["val", "test"]:
                m = compute_metrics(predict(model, data[split], prep), data[split], prep)
                rows.append({"model": name, "seed": seed, "split": split,
                             "best_epoch": int(history["val_score"].idxmin()) + 1, **m})
            print(f"done: {name}, seed {seed}")
    results = pd.DataFrame(rows)
    results.to_csv(METRICS_DIR / "dl_seed_comparison.csv", index=False)
    return results


if __name__ == "__main__":
    main()
