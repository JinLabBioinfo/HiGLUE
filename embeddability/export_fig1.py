r"""
Writes the Figure 1b inputs into paper/figures from embeddability/results:

* fig1_cluster_then_name.csv   per dataset: break-even contamination, HiGLUE / Hi-C-only / Fast-Higashi / perfect-cluster
                               balanced accuracy, real-cluster impurity, median contacts, types (in-house islet left out)
* fig1_droplet_degradation.csv Droplet Hi-C cortex: balanced accuracy of naming perfect clusters per contamination p
                               and mode (mean, min, max over repeats)
* fig1_droplet_real.csv        Droplet Hi-C cortex: impurity and re-named balanced accuracy of the real standalone
                               clusterings (Hi-C-only, Fast-Higashi, scGAD-only), per method and seed

Run after cluster_degradation.py, scgad_alone.py and plot_degradation.py: python embeddability/export_fig1.py
"""

import pandas as pd

RES = "embeddability/results"
OUT = "paper/figures"

s = pd.read_csv(f"{RES}/degradation_summary.csv").set_index("dataset")
v = pd.read_csv(f"{RES}/supp_embeddability_values.csv").set_index("dataset")
t = s.join(v[["median_depth", "n_types", "bacc_perfect_clusters"]])
t = t.rename(columns={"higlue": "bacc_higlue", "bacc_perfect_clusters": "bacc_perfect"}).drop(columns=["ceiling"])
# scGAD-only clusters (scgad_alone.py): the scHi-C gene scores embedded and clustered without the RNA, named the same way
g = pd.read_csv(f"{RES}/scgad_alone.csv").groupby("dataset")
t["impurity_scgad_alone"] = g.impurity.mean()
t["bacc_scgad_alone"] = g.balanced_accuracy_renamed.mean()
t["knn_scgad_alone"] = g.knn_bacc.mean()
t.drop(index="islet_inhouse").round(4).to_csv(f"{OUT}/fig1_cluster_then_name.csv")

d = pd.read_csv(f"{RES}/degradation_impurity.csv")
d = d[d.dataset == "droplet_cortex"].groupby(["mode", "p"]).balanced_accuracy.agg(["mean", "min", "max"])
d.round(4).to_csv(f"{OUT}/fig1_droplet_degradation.csv")

r = pd.read_csv(f"{RES}/real_cluster_impurity.csv")
r = pd.concat([r, pd.read_csv(f"{RES}/scgad_alone.csv")])
r = r[r.dataset == "droplet_cortex"][["method", "seed", "impurity", "n_clusters", "balanced_accuracy_renamed"]]
r.round(4).to_csv(f"{OUT}/fig1_droplet_real.csv", index=False)
print("wrote", OUT)
