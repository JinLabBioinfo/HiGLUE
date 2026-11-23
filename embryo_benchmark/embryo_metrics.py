r"""
Full HiRES mouse embryo benchmark (hires_embryo_full; 7 stages, 7,154 labelled scHi-C cells): metrics beyond the
21-type label transfer of benchmark_main.sh, on the scHi-C cells of every method and seed.

1. Stage: label-transfer accuracy and balanced accuracy per developmental stage.
2. Lineage: the 21 RNA cell types folded into 8 lineages (LINEAGE); accuracy and balanced accuracy of the folded
   predictions. Cells annotated "mitosis" (a cell-cycle state, not a lineage) and "Unknown" are left out as truth; a
   prediction of "mitosis" for any other cell counts as wrong.
3. Cell cycle (phase of the paired RNA cell, HiRES annotation; G0 left out; circular order G1, Early-S, Mid-S, Late-S,
   G2, M), on each method's scHi-C embedding:
   - ACROC, average circular ROC (after Liu et al. 2018, Bioinformatics): the embedding's first two PCs, each cell's
     angle around the centre; per phase, the ROC AUC of phase vs rest with the angle as score, at the best rotation of
     the starting point (a phase on one contiguous arc scores 1); averaged over the phases. Reported with the same
     score for shuffled phases (the best rotation inflates chance above 0.5).
   - kNN phase score: mean circular distance (in phase steps, 0-3) between a cell's phase and those of its 15 nearest
     scHi-C neighbours, as 1 - distance / shuffled distance (0 = chance, 1 = neighbours always share the phase).
   - Phase transfer (joint embeddings only): each scHi-C cell's phase by majority of its 15 nearest RNA cells;
     accuracy and mean circular error.
Methods: HiGLUE; Fast-Higashi (embedding; its GM labels for 1-2); Fast-Higashi-smoothed scGAD + Harmony; scGAD +
Harmony; scGAD + Seurat CCA; band PCs + GM (labels only, no saved embedding).
Output: embryo_benchmark/embryo_metrics[/<dataset>]/{stage,lineage,cycle}.csv and summary.txt
Run (repository root, envs/higlue): python embryo_benchmark/embryo_metrics.py [--dataset hires_embryo_full_nomito]
"""
import argparse
import glob
import os
import warnings

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")
DS = "hires_embryo_full"
UNL = {"Unknown", "Other", "nan", "NA", "None"}
STAGES = ["E70", "E75", "E80", "E85", "E95", "EX05", "EX15"]
CYCLE = ["G1", "Early-S", "Mid-S", "Late-S", "G2", "M"]
K = 15
# 21 RNA types -> 8 lineages. Epithelial cells express Trp63 / Tfap2a / Wnt6 with little Epcam / Krt8 (surface
# ectoderm); NMP (T, Cdx2, Hoxb9) are axial progenitors, kept with the epiblast / primitive streak; Schwann cell
# precursors (Sox10) are neural crest, with the neural lineage; notochord is axial mesoderm.
LINEAGE = {
    "epiblast and PS": "Epiblast, PS & NMP", "NMP": "Epiblast, PS & NMP",
    "neural ectoderm": "Neural", "neural tube": "Neural", "radial glias": "Neural", "early neurons": "Neural",
    "oligodendrocytes and progenitors": "Neural", "schwann cell precursors": "Neural",
    "epithelial cells": "Surface ectoderm",
    "early mesoderm": "Mesoderm & mesenchyme", "intermediate mesoderm": "Mesoderm & mesenchyme",
    "early mesenchyme": "Mesoderm & mesenchyme", "mix late mesenchyme": "Mesoderm & mesenchyme",
    "myocytes": "Mesoderm & mesenchyme", "notochord": "Mesoderm & mesenchyme",
    "blood": "Blood", "endoderm": "Endoderm",
    "ExE ectoderm": "Extra-embryonic", "ExE endoderm": "Extra-embryonic", "ExE mesoderm": "Extra-embryonic",
    "mitosis": "mitosis",
}
METHODS = {  # name -> (label source, embedding source); None = not available
    "HiGLUE": ("higlue", "higlue"),
    "Fast-Higashi + GM": ("fasthigashi_gm", "fasthigashi"),
    "Fast-Higashi-smoothed scGAD + Harmony": ("fasthigashi_harmony", "fasthigashi_harmony"),
    "scGAD + Harmony": ("scgad", "scgad"),
    "scGAD + Seurat CCA": ("scgad_cca", "scgad_cca"),
    "Band PCs + GM": ("graph_match", None),
}


def labels(sd, src):
    if src in ("graph_match", "fasthigashi_gm"):
        f = f"{sd}/eval/baselines/{src}/predictions.csv"
        if not os.path.exists(f):
            return None
        p = pd.read_csv(f, index_col=0)
        return pd.DataFrame({"pred": p["pred"].astype(str), "truth": p["truth"].astype(str)})
    o = obs_of(sd, src)
    if o is None:
        return None
    o = o[o["domain"] == "hic"]  # paired data: RNA, GAD and Hi-C cells share names
    o = o[~o.index.duplicated()]
    return pd.DataFrame({"pred": o["pred_celltype_harmonized"].astype(str), "truth": o["celltype"].astype(str)})


def path_of(sd, src):
    return f"{sd}/eval/embeddings.h5ad" if src == "higlue" else f"{sd}/eval/baselines/{src}/embeddings.h5ad"


def obs_of(sd, src):
    f = path_of(sd, src)
    if not os.path.exists(f):
        return None
    return ad.read_h5ad(f, backed="r").obs


def embedding(sd, src):
    r"""scHi-C and (joint methods) RNA cells of a method's embedding"""
    f = path_of(sd, src)
    if not os.path.exists(f):
        return None, None
    a = ad.read_h5ad(f)
    key = [k for k in a.obsm if k != "X_umap"][0]
    if "domain" not in a.obs:  # Hi-C-only embedding (Fast-Higashi)
        return pd.DataFrame(a.obsm[key], index=a.obs_names), None
    X, dom = a.obsm[key], a.obs["domain"].astype(str).to_numpy()
    hic = pd.DataFrame(X[dom == "hic"], index=a.obs_names[dom == "hic"])
    rna = pd.DataFrame(X[dom == "rna"], index=a.obs_names[dom == "rna"])
    return hic[~hic.index.duplicated()], rna[~rna.index.duplicated()]


def acc_bacc(t, p):
    return float((t == p).mean()), float(balanced_accuracy_score(t, p))


# ------------------------------------------------------------------ cell cycle
def acroc(xy, phase):
    r"""Average circular ROC of a 2D embedding: per phase the best-rotation AUC of phase vs rest by angle"""
    c = xy - xy.mean(0)
    order = np.argsort(np.arctan2(c[:, 1], c[:, 0]), kind="stable")
    ph = np.asarray(phase)[order]
    n, i = len(ph), np.arange(len(ph))
    aucs = []
    for p in CYCLE:
        y = ph == p
        npos = int(y.sum())
        if npos == 0 or npos == n:
            continue
        # start at index s: rank of position j = (j - s) mod n; sum over positives for every s at once
        before = np.concatenate([[0], np.cumsum(y)[:-1]])  # positives at positions < s
        rank_sum = i[y].sum() - np.arange(n) * npos + n * before
        auc = (rank_sum - npos * (npos - 1) / 2) / (npos * (n - npos))
        aucs.append(auc.max())
    return float(np.mean(aucs))


def circ(a, b):
    d = np.abs(np.asarray(a) - np.asarray(b)) % len(CYCLE)
    return np.minimum(d, len(CYCLE) - d)


def knn_phase(X, code, rng):
    idx = NearestNeighbors(n_neighbors=K + 1).fit(X).kneighbors(X, return_distance=False)[:, 1:]
    d = circ(code[:, None], code[idx]).mean()
    d0 = np.mean([circ(code[:, None], rng.permutation(code)[idx]).mean() for _ in range(5)])
    return 1 - d / d0


def phase_transfer(hic, rna, ph_hic, ph_rna):
    idx = NearestNeighbors(n_neighbors=K).fit(rna).kneighbors(hic, return_distance=False)
    nb = ph_rna[idx]
    pred = np.array([np.bincount(r, minlength=len(CYCLE)).argmax() for r in nb])
    return float((pred == ph_hic).mean()), float(circ(pred, ph_hic).mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=DS)
    ds = p.parse_args().dataset
    RUNS = f"cluster_bundle/higlue_bundle/main_runs/{ds}/500kb_genebody_gadcw"
    OUT = "embryo_benchmark/embryo_metrics" + ("" if ds == DS else f"/{ds}")
    os.makedirs(OUT, exist_ok=True)
    ref = pd.read_csv(f"datasets/{ds}/hic_ref.tsv", sep="\t", index_col=0)
    rna_phase = ad.read_h5ad(f"datasets/{ds}/rna.h5ad", backed="r").obs["cellcycle_phase"].astype(str)
    rng = np.random.default_rng(0)
    st_rows, li_rows, cy_rows = [], [], []
    for sd in sorted(glob.glob(f"{RUNS}/seed*")):
        seed = int(sd.rsplit("seed", 1)[1])
        for m, (lsrc, esrc) in METHODS.items():
            lab = labels(sd, lsrc)
            if lab is not None:
                lab = lab[~lab.index.duplicated() & ~lab["truth"].isin(UNL)]
                stage = ref["stage"].reindex(lab.index)
                st_rows.append({"method": m, "seed": seed, "stage": "all", "n": len(lab),
                                **dict(zip(("accuracy", "balanced_accuracy"), acc_bacc(lab.truth, lab.pred)))})
                for s in STAGES:
                    x = lab[stage == s]
                    st_rows.append({"method": m, "seed": seed, "stage": s, "n": len(x),
                                    **dict(zip(("accuracy", "balanced_accuracy"), acc_bacc(x.truth, x.pred)))})
                x = lab[lab["truth"] != "mitosis"]
                t, p = x["truth"].map(LINEAGE), x["pred"].map(LINEAGE).fillna("other")
                assert t.notna().all(), set(x.truth[t.isna()])
                a, b = acc_bacc(t, p)
                r = {"method": m, "seed": seed, "n": len(x), "accuracy": a, "balanced_accuracy": b,
                     "type_accuracy_same_cells": float((x.truth == x.pred).mean()),
                     "pred_mitosis_share": float((x.pred == "mitosis").mean())}
                r.update({f"recall_{g}": float((p[t == g] == g).mean()) for g in sorted(set(t))})
                li_rows.append(r)
            if esrc is None:
                continue
            hic, rna = embedding(sd, esrc)
            if hic is None:
                continue
            ph = ref["phase"].reindex(hic.index).astype(str)
            keep = ph.isin(CYCLE).to_numpy()
            Xh, code = hic.to_numpy()[keep], ph[keep].map({p: i for i, p in enumerate(CYCLE)}).to_numpy()
            xy = PCA(n_components=2, random_state=0).fit_transform(Xh)
            r = {"method": m, "seed": seed, "n": int(keep.sum()), "acroc": acroc(xy, ph[keep]),
                 "acroc_shuffled": float(np.mean([acroc(xy, rng.permutation(ph[keep].to_numpy())) for _ in range(5)])),
                 "knn_phase_score": knn_phase(Xh, code, rng)}
            if rna is not None:
                pr = rna_phase.reindex(rna.index)
                kr = pr.isin(CYCLE).to_numpy()
                acc, err = phase_transfer(Xh, rna.to_numpy()[kr], code,
                                          pr[kr].map({p: i for i, p in enumerate(CYCLE)}).to_numpy())
                r.update({"phase_transfer_accuracy": acc, "phase_transfer_circular_error": err})
            cy_rows.append(r)
            print(f"seed {seed} {m}: " + ", ".join(f"{k} {v:.3f}" for k, v in r.items() if isinstance(v, float)),
                  flush=True)
    st, li, cy = pd.DataFrame(st_rows), pd.DataFrame(li_rows), pd.DataFrame(cy_rows)
    for name, d in (("stage", st), ("lineage", li), ("cycle", cy)):
        d.to_csv(f"{OUT}/{name}.csv", index=False)
    order = list(METHODS)
    fmt = lambda x: f"{x.mean():.3f} ± {x.std():.3f}" if len(x) > 1 else f"{x.mean():.3f}"  # noqa: E731
    lines = ["== stage accuracy (mean of seeds)",
             st.pivot_table(index="method", columns="stage", values="accuracy")[STAGES + ["all"]].reindex(order)
             .round(3).to_string(),
             "\n== stage balanced accuracy",
             st.pivot_table(index="method", columns="stage", values="balanced_accuracy")[STAGES + ["all"]]
             .reindex(order).round(3).to_string(),
             "\n== lineage (8 lineages, mitosis cells excluded)",
             li.groupby("method")[["accuracy", "balanced_accuracy", "type_accuracy_same_cells", "pred_mitosis_share"]]
             .agg(fmt).reindex(order).to_string(),
             li.groupby("method")[[c for c in li if c.startswith("recall_")]].mean().reindex(order).round(2)
             .T.to_string(),
             "\n== cell cycle (scHi-C cells, G0 excluded)",
             cy.groupby("method")[[c for c in cy if c not in ("method", "seed", "n")]].agg(fmt).reindex(order)
             .dropna(how="all").to_string(),
             "\nseeds per method: " + li.groupby("method").size().to_dict().__repr__()]
    txt = "\n".join(lines)
    open(f"{OUT}/summary.txt", "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
