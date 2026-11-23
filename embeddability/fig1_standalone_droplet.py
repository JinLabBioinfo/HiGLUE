r"""
Figure 1c: two more scHi-C-standalone embeddings of the Droplet Hi-C cortex, next to Fast-Higashi (user, 2026-11-18)

  InnerProduct  scHiCTools inner-product embedding (K = 10 diagonals, the better of K = 10 / 50; embeddability.py,
                results/innerproduct_droplet_cortex.npz)
  scGAD alone   the scGAD gene scores of the scHi-C cells without the RNA (scgad.integrate's ``hic_alone``, 30 PCs)
Each is clustered and named exactly like the other cluster-then-name baselines (scgad_alone.auto_clusters: graph
matching's resolution scan; cluster_degradation.name: reference types with >= MIN_REF cells), seeds 1-3 (Leiden; and
PCA for scGAD). scGAD-alone accuracies are recomputed here, so they can be compared with results/scgad_alone.csv.
UMAP of seed 1 (default UMAP parameters, random_state 0, as Fast-Higashi in make_fig1.py).

Writes paper/figures/fig1_standalone_cache.npz ({ip,gad}_xy, {ip,gad}_ct, {ip,gad}_bacc (3 seeds)).
Run (repository root, envs/higlue): python embeddability/fig1_standalone_droplet.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import evals  # noqa: E402
import graph_match as gm  # noqa: E402
import scgad  # noqa: E402
from cluster_degradation import MIN_REF, name, score  # noqa: E402
from scgad_alone import auto_clusters, load  # noqa: E402

DS = "droplet_cortex"
OUT = os.path.join(os.path.dirname(HERE), "paper", "figures", "fig1_standalone_cache.npz")


def main():
    import umap
    rna, hic, counts, graph = load(DS)
    truth = hic.obs.celltype.astype(str).to_numpy()
    label_map = evals.get_label_map(DS, truth, rna.obs["celltype"])
    rna_sig = gm.rna_signatures(rna, evals.harmonize(rna.obs["celltype"], label_map), min_cells=MIN_REF)
    edges = gm.overlap_edges(graph)
    x = gm.hic_counts(hic)
    z = np.load(os.path.join(HERE, "results", f"innerproduct_{DS}.npz"), allow_pickle=True)
    ip = pd.DataFrame(z["s10"], index=z["cells"].astype(str)).reindex(hic.obs_names.astype(str))
    assert not ip.isna().any().any(), "InnerProduct cells missing"
    out = {}
    for key in ("ip", "gad"):
        acc, emb1 = [], None
        for seed in (1, 2, 3):
            emb = ip.to_numpy() if key == "ip" else scgad.integrate(rna, counts, seed=seed)["hic_alone"]
            res = auto_clusters(emb, hic, x, rna_sig, edges, seed)
            cl = res["clusters"].astype(str).to_numpy().astype(object)
            acc.append(score(truth, cl, name(cl, hic, x, rna_sig, edges).to_dict())["balanced_accuracy"])
            print(key, seed, round(acc[-1], 4), "resolution", res["resolution"], flush=True)
            if seed == 1:
                emb1 = np.asarray(emb)
        out[f"{key}_xy"] = umap.UMAP(random_state=0).fit_transform(emb1)
        out[f"{key}_ct"] = truth
        out[f"{key}_bacc"] = np.array(acc)
    np.savez(OUT, **out)
    print("wrote", OUT, {k: v.round(3) for k, v in out.items() if k.endswith("bacc")})


if __name__ == "__main__":
    main()
