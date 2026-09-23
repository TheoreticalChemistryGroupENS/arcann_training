"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2022-2024 ArcaNN developers group <https://github.com/arcann-chem>                     #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2025/12/19
Last modified: 2025/12/19
"""

import json
import logging
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

arcann_logger = logging.getLogger("ArcaNN")


def _parse_mace_results(results_file: Path):
    """Parse a MACE JSONL results file into opt and eval record lists."""
    opt_records, eval_records = [], []
    with results_file.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("mode") == "opt":
                opt_records.append(record)
            elif record.get("mode") == "eval":
                eval_records.append(record)
    return opt_records, eval_records


def plot_mace_learning_curves(results_file: Path) -> None:
    """
    Parse a MACE training results file and save a 2x3 learning-curve figure
    next to it in the same directory.

    The figure shows, as a function of epoch (eval checkpoints):
      - Total loss  (+ smoothed opt-step loss in background)
      - Energy RMSE per atom
      - Force RMSE
      - Energy MAE per atom
      - Force MAE
      - Force relative RMSE

    Parameters
    ----------
    results_file : Path
        Path to the MACE JSONL results file (e.g. model_1_000_run-*_train.txt).
    """
    opt_records, eval_records = _parse_mace_results(results_file)

    # Drop the pre-training evaluation entry (epoch is None)
    eval_records = [r for r in eval_records if r.get("epoch") is not None]
    if not eval_records:
        arcann_logger.warning(
            f"No epoch-tagged eval records in {results_file}, skipping plot."
        )
        return

    epochs = np.array([r["epoch"] for r in eval_records])
    loss_eval = np.array([r["loss"] for r in eval_records])
    rmse_e = np.array([r["rmse_e_per_atom"] for r in eval_records])
    mae_e = np.array([r["mae_e_per_atom"] for r in eval_records])
    rmse_f = np.array([r["rmse_f"] for r in eval_records])
    mae_f = np.array([r["mae_f"] for r in eval_records])
    rel_rmse_f = np.array([r["rel_rmse_f"] for r in eval_records])

    opt_loss = np.array([r["loss"] for r in opt_records]) if opt_records else None

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        f"MACE training – {results_file.parent.parent.name} / {results_file.name}",
        fontsize=11,
    )

    color_valid = "#2C73D2"
    color_train = "#BBBBBB"
    color_mae = "#E84393"
    color_rel = "#FF6B35"

    def _add_panel(ax, x, y, color, ylabel, title, marker="o"):
        if opt_loss is not None and title == "Total loss":
            window = max(1, len(opt_loss) // 200)
            smoothed = np.convolve(opt_loss, np.ones(window) / window, mode="valid")
            x_opt = np.linspace(x[0], x[-1], len(smoothed))
            ax.plot(
                x_opt,
                smoothed,
                color=color_train,
                lw=0.8,
                label="train (batch, smoothed)",
            )
        ax.plot(
            x, y, color=color, lw=1.5, marker=marker, ms=3, label="valid (per epoch)"
        )
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.4)
        if title == "Total loss" and opt_loss is not None:
            ax.legend(fontsize=8)

    _add_panel(
        axes[0, 0], epochs, loss_eval, color_valid, "Loss", "Total loss (train + valid)"
    )
    _add_panel(
        axes[0, 1],
        epochs,
        rmse_e,
        color_valid,
        "RMSE [eV/atom]",
        "Energy RMSE per atom (valid)",
    )
    _add_panel(
        axes[0, 2], epochs, rmse_f, color_valid, "RMSE [eV/Å]", "Force RMSE (valid)"
    )
    _add_panel(
        axes[1, 0],
        epochs,
        mae_e,
        color_mae,
        "MAE [eV/atom]",
        "Energy MAE per atom (valid)",
        marker="s",
    )
    _add_panel(
        axes[1, 1],
        epochs,
        mae_f,
        color_mae,
        "MAE [eV/Å]",
        "Force MAE (valid)",
        marker="s",
    )
    _add_panel(
        axes[1, 2],
        epochs,
        rel_rmse_f,
        color_rel,
        "Rel. RMSE [%]",
        "Force relative RMSE (valid)",
        marker="^",
    )

    plt.tight_layout()
    out_path = results_file.parent / (results_file.stem + "_lcurve.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    arcann_logger.info(f"Learning curve saved to {out_path}")
