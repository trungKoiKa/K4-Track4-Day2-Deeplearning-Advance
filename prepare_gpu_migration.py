"""Create an auditable CPU-to-T4 resume bundle after a completed epoch.

This tool never edits the active CPU study.  It copies one saved run to a
separate staging directory, preserves the original CPU metadata, and writes a
GPU-resume metadata record.  The generated ZIP contains checkpoints and is
intentionally ignored by Git: upload it only to the user's Google Drive before
resuming in Colab.
"""
from __future__ import annotations

import argparse
import shutil
import zipfile
from dataclasses import asdict
from pathlib import Path

import torch

from lab_solution.common import read_json, sha256, write_json
from lab_solution.train import Config, compatibility


REQUIRED = (
    "best.pt", "last.pt", "config.json", "metadata.json", "environment.json",
    "pretrained.json", "profile.json", "split_checks.json", "history.csv",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-root", type=Path, required=True)
    parser.add_argument("--exp-id", default="F01")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--after-epoch", type=int, required=True,
                        help="Only bundle after this epoch has been checkpointed.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Ignored ZIP destination, outside the Git repository.")
    args = parser.parse_args()

    source = args.cpu_root.resolve() / "runs" / args.exp_id / f"seed{args.seed}"
    missing = [name for name in REQUIRED if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete source run {source}: {missing}")
    state = torch.load(source / "last.pt", map_location="cpu", weights_only=False)
    saved_epoch = int(state["epoch"]) + 1
    if saved_epoch < args.after_epoch:
        raise RuntimeError(
            f"last.pt contains epoch {saved_epoch}; wait until epoch {args.after_epoch} is saved."
        )

    original_cfg = read_json(source / "config.json")
    gpu_cfg = {**original_cfg, "device": "cuda", "amp": False, "resume": True}
    source_meta = read_json(source / "metadata.json")
    expected_meta = {
        "compatibility": compatibility(Config(**gpu_cfg)),
        "data_hashes": source_meta["data_hashes"],
        "code_hashes": source_meta["code_hashes"],
    }
    staging = args.output.resolve().with_suffix("").parent / f"{args.exp_id}_seed{args.seed}_gpu_resume"
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(source, staging)
    write_json(staging / "config_before_gpu_migration.json", original_cfg)
    write_json(staging / "metadata_before_gpu_migration.json", source_meta)
    write_json(staging / "config.json", gpu_cfg)
    write_json(staging / "metadata.json", expected_meta)
    record = {
        "source": str(source), "exp_id": args.exp_id, "seed": args.seed,
        "checkpointed_epoch": saved_epoch, "from_device": "cpu", "to_device": "cuda",
        "amp": False,
        "reason": "GPU migration preserves optimizer, scheduler, RNG and run identity; AMP remains disabled for compatible continuation.",
        "source_last_sha256": sha256(source / "last.pt"),
        "source_best_sha256": sha256(source / "best.pt"),
        "source_config": original_cfg,
        "gpu_config": gpu_cfg,
        "gpu_metadata": expected_meta,
    }
    write_json(staging / "hardware_migration.json", record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in staging.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(staging.parent))
    print({"bundle": str(args.output), "staging": str(staging), "saved_epoch": saved_epoch,
           "last_sha256": record["source_last_sha256"]})


if __name__ == "__main__":
    main()
