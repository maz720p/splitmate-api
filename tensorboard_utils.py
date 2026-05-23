"""
tensorboard_utils.py  -  SplitMate TensorBoard Log Reader
==========================================================
Utility untuk membaca metrik training dari TensorBoard event files.
Digunakan oleh expense_api.py untuk endpoint /training-metrics.

Mendukung 4 split dari training v4:
  train / validation       -> dari model.fit (Functional API)
  gt_train / gt_val        -> dari tf.GradientTape (Subclassing)
"""

import os
from typing import Dict, List, Optional, Any
import tensorflow as tf

LOG_DIR = os.environ.get("LOGS_DIR", "logs")

# Semua split yang mungkin ada di log v4
ALL_SPLITS = ["train", "validation", "gt_train", "gt_val"]


def _parse_run(run_path: str) -> Dict[str, Dict[str, List[Dict]]]:
    """
    Baca semua event files dalam satu run directory.
    Return: {split: {tag: [{step, value}]}}
    """
    result: Dict[str, Dict[str, List[Dict]]] = {}

    for split in ALL_SPLITS:
        split_path = os.path.join(run_path, split)
        if not os.path.isdir(split_path):
            continue

        result[split] = {}
        for fname in sorted(os.listdir(split_path)):
            if "tfevents" not in fname:
                continue
            fpath = os.path.join(split_path, fname)
            try:
                for event in tf.compat.v1.train.summary_iterator(fpath):
                    step = int(event.step)
                    for v in event.summary.value:
                        tag = v.tag
                        if "histogram" in tag or tag == "keras":
                            continue
                        try:
                            val = float(tf.make_ndarray(v.tensor).item())
                        except Exception:
                            val = float(v.simple_value)
                        if tag not in result[split]:
                            result[split][tag] = []
                        result[split][tag].append({"step": step, "value": val})
            except Exception:
                continue

        for tag in result[split]:
            result[split][tag].sort(key=lambda x: x["step"])

    return result


def list_runs() -> List[str]:
    """Daftar nama run yang tersedia di LOG_DIR."""
    if not os.path.isdir(LOG_DIR):
        return []
    return sorted([
        d for d in os.listdir(LOG_DIR)
        if os.path.isdir(os.path.join(LOG_DIR, d))
    ])


def get_run_metrics(run_name: str) -> Optional[Dict[str, Any]]:
    """
    Ambil semua metrik untuk satu run.
    Return None jika run tidak ditemukan.
    """
    run_path = os.path.join(LOG_DIR, run_name)
    if not os.path.isdir(run_path):
        return None

    data = _parse_run(run_path)

    summary: Dict[str, Dict[str, Dict]] = {}
    for split, tags in data.items():
        summary[split] = {}
        for tag, vals in tags.items():
            values = [v["value"] for v in vals]
            if values:
                summary[split][tag] = {
                    "n_steps": len(values),
                    "first"  : round(values[0],  8),
                    "last"   : round(values[-1],  8),
                    "min"    : round(min(values),  8),
                    "max"    : round(max(values),  8),
                }

    return {
        "run_name": run_name,
        "splits"  : list(data.keys()),
        "summary" : summary,
        "series"  : data,
    }


def get_all_runs_summary() -> List[Dict[str, Any]]:
    """Ringkasan semua run tanpa series lengkap."""
    runs = []
    for run_name in list_runs():
        metrics = get_run_metrics(run_name)
        if metrics:
            runs.append({
                "run_name": metrics["run_name"],
                "splits"  : metrics["splits"],
                "summary" : metrics["summary"],
            })
    return runs
