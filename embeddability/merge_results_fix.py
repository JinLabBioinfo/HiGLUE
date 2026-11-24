r"""Merge embeddability/results_fix into embeddability/results (2026-11-24).

results_fix holds the embeddability, perfect-cluster and cluster-then-name analyses rerun for the datasets whose runs
changed (band-fixed HiRES brain and Droplet Hi-C cortex, islet atlas rerun; cluster_bundle/paper_runs). For every CSV
present in both folders, the rows of those datasets are replaced and all other rows kept; the previous file is saved
once as <name>.before_fix.csv.
Usage (repository root): python embeddability/merge_results_fix.py
"""
import os
import shutil

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FIX, RES = os.path.join(HERE, "results_fix"), os.path.join(HERE, "results")


def main():
    for name in sorted(os.listdir(FIX)):
        if not name.endswith(".csv") or not os.path.exists(os.path.join(RES, name)):
            continue
        new = pd.read_csv(os.path.join(FIX, name))
        old_path = os.path.join(RES, name)
        old = pd.read_csv(old_path)
        if "dataset" not in new or "dataset" not in old:
            print(f"skip {name}: no dataset column")
            continue
        backup = old_path.replace(".csv", ".before_fix.csv")
        if not os.path.exists(backup):
            shutil.copy(old_path, backup)
        ds = set(new["dataset"])
        out = pd.concat([old[~old["dataset"].isin(ds)], new], ignore_index=True)
        out = out[list(old.columns) + [c for c in out.columns if c not in old.columns]]
        out.to_csv(old_path, index=False)
        print(f"{name}: replaced {int(old['dataset'].isin(ds).sum())} rows of {sorted(ds)} with {len(new)}")


if __name__ == "__main__":
    main()
