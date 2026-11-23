# HiGLUE
Joint generative modeling of single-cell Hi-C with other single-cell modalities via Graph-Linked Unified Embedding (GLUE)

HiGLUE is an extension of [GLUE](https://scglue.readthedocs.io/en/latest/) for scHi-C. It embeds scHi-C and scRNA-seq (optionally scATAC-seq and DNA methylation) into one latent space, guided by a graph that links genes to the Hi-C bins around them, and transfers RNA cell type labels to the Hi-C cells.

## Installation

HiGLUE uses a modified copy of `scglue` that lives in this repository (`scglue/`). Install it from there rather than from PyPI or a GLUE fork:

```bash
git clone <this repository> HiGLUE && cd HiGLUE
conda create -n higlue python=3.9 && conda activate higlue
# PyTorch for your CUDA version first, see https://pytorch.org
pip install ./scglue                                        # the local scglue
pip install git+https://github.com/JinLabBioinfo/SCORE.git  # scHi-C loading and preprocessing
pip install local-attention cooltools wandb                 # attention decoder, structure analysis, logging
pip install harmonypy                                       # scGAD baseline (--baselines scgad)
```

After changing anything under `scglue/`, reinstall it into the environment (without touching the other packages) with:

```bash
bash reinstall.sh   # = cd scglue && pip install . --no-dependencies
```

`higlue.py` imports `scglue` from the environment, not from the folder, so edits only take effect after reinstalling.

## Inputs

* **RNA**: an `.h5ad` with raw counts (in `layers["counts"]` or `X`), cell type labels in `obs["celltype"]` and, optionally, `obs["batch"]`.
* **scHi-C**: a `.scool` of single-cell coolers at the resolution you want to model (e.g. 500kb).
* **scHi-C metadata** (`--reference`): a tab-separated file with one row per cell and the columns `cell` (name in the `.scool`), `depth`, `batch` and `cluster` (cell type; use `Unknown` for unlabeled cells):

  ```
  cell          depth    trans   total    batch  cluster
  ValbB8w1074   1591802  664342  2256144  0      In2
  ```
* **Genes**: a `.gtf` of the genome assembly the RNA and Hi-C were mapped to (it places genes on the Hi-C bins).

## Running HiGLUE

A minimal run (as in `hires_minimal.sh`):

```bash
python higlue.py \
    --rna_file hires_brain_rna.h5ad \
    --gtf gencode.vM25.annotation.gtf \
    --train \
    --preprocess \
    SCORE \
    --dset hires_brain \
    --scool hires_brain_500kb.scool \
    --reference hires_brain_ref \
    --resolution 500kb \
    --min_depth 100000
```

* Arguments before `SCORE` configure HiGLUE; arguments after it load and filter the scHi-C data with SCORE (`--dset` names the dataset, `--resolution` must match the `.scool`, `--min_depth` drops shallow cells).
* `--preprocess` builds the RNA and Hi-C features and the guidance graph in `data/<dset>_data/` (or `--data_dir`); `--train` trains on them. Pass both, or preprocess once and train repeatedly. Change the preprocessing (e.g. `--n_genes`, `--n_strata`, `--loop_q`) and you need to preprocess again.
* Give every run its own `--exp_name`: training checkpoints go to `<exp_name>/` and the evaluation to `<exp_name>/eval` (both default to `glue/`, which the next run would overwrite). The trained model is saved to `data/<dset>_data/` and, with `--cache_checkpoint <dir>`, also to `<dir>/`.

### Common options

| Option | Default | Meaning |
| --- | --- | --- |
| `--n_genes` | 10000 | highly variable genes used from the RNA |
| `--n_strata` | 10 | distance bands (diagonals) of the Hi-C matrix used as features; with `--n_strata 10` at 500kb the model sees contacts up to 5Mb apart |
| `--loop_q` | 0.98 | quantile of pseudobulk contacts that become Hi-C edges of the guidance graph |
| `--use_dist_norm` | off | rank contacts for the graph after distance (observed/expected) normalization |
| `--use_trans` | off | also add trans-chromosomal contact edges to the graph |
| `--use_batch batch` | off | correct the RNA for `obs["batch"]` (Hi-C batches from the reference are always used) |
| `--batch_size` | 128 | minibatch size |
| `--lam_align` | 0.01 | weight of the adversarial alignment between modalities |
| `--hic_weight` | 10 | weight of the Hi-C reconstruction relative to RNA |
| `--balance` | off | reweight cells so that cell type proportions need not match between modalities |
| `--celltypes A B ...` | all | train and evaluate on these cell types only, in every modality |
| `--gene_list G1 G2 ...` | none | genes whose predicted expression is drawn on the joint UMAP after training |
| `--seed` | 25 | random seed |
| `--max_epochs` | auto | cap training epochs (default: early stopping) |
| `--wandb` | off | log training and evaluation metrics to Weights & Biases |

`python higlue.py --help SCORE` lists all options.

**Choosing settings.** When the RNA reference and the Hi-C cells have similar cell type compositions (e.g. RNA and Hi-C from the same cells or tissue samples), the defaults work well. When they do not, the alignment tends to force the modalities to fully mix and assigns surplus Hi-C cells to the most abundant RNA type, and results vary between seeds. A weaker alignment (e.g. `--lam_align 0.002`) usually helps; restricting the RNA to the cell types present in the Hi-C (`--celltypes`) also helps. Compare a few seeds before trusting a configuration.

## Evaluation after training

Every training run ends with an evaluation (`evals.py`) written to `--eval_dir` (default `<exp_name>/eval`):

* RNA labels are transferred to the Hi-C cells (`obs["pred_celltype"]`; the annotation in `obs["celltype"]` is kept as ground truth). If the RNA and Hi-C annotations use different names for the same types, a label map in `evals.DATASET_CONFIGS` translates them before scoring.
* `metrics.csv`: accuracy, balanced accuracy, macro F1, label transfer ARI/NMI, Leiden ARI and cell type silhouette of each modality, modality mixing, and, for paired cells, FOSCTTM. Cells labeled `Unknown` are predicted but not scored; when only some cells are labeled, label metrics are reported as `labeled_*`.
* Plots: Hi-C UMAP (truth, prediction, confidence, depth, batch), joint UMAPs, and a confusion matrix of true vs transferred labels; `per_class_hic.csv` (recall per type) and `pred_composition_hic.csv`.
* `embeddings.h5ad`: the joint embedding and annotations of all cells, for later analysis.

With `--wandb` the metrics and plots are logged too.

### Re-evaluating a trained model

```bash
# from the embeddings saved at the end of training
python evals.py embeddings <eval_dir>/embeddings.h5ad --out_dir <dir>

# from a saved model and the preprocessed data it was trained on
python evals.py model <model>.dill \
    --rna data/<dset>_data/rna/rna_<suffix>.h5ad \
    --hic data/<dset>_data/hic/hic_<suffix>.h5ad \
    --min_depth 100000 --out_dir <dir>
```

Both accept `--dset` (selects the label map) and `--celltypes`.

## Chromatin structure analysis

Single Hi-C cells are too sparse to read structure at a gene, but cells pooled by cell type are not. `--structure_eval` pools the Hi-C cells by predicted type (and by their annotation, where cells are labeled) into pseudobulk coolers and uses cooltools to compute insulation and A/B compartments (E1) of every pool:

```bash
python higlue.py ... --train \
    --structure_eval \
    --structure_scool hires_brain_100kb.scool \
    SCORE ...
```

(see `hires_minimal_struct.sh`). Results go to `<eval_dir>/structure/`:

| Output | What it shows |
| --- | --- |
| `<grouping>_markers.png`, `_tracks.png` | insulation and E1 of every pool at marker genes, and tracks ±2Mb around one marker per type. The diagonal score asks whether a type's markers sit in more A / differently insulated chromatin in that type's own pool (p-value from random genes) |
| `<grouping>_<feature>_specific_loci.png` | loci found from the Hi-C alone where one pool is most A (E1) or most insulated relative to the others, next to the RNA expression of their genes (graph-linked genes and genes with their TSS in the locus) per RNA type. The concordance score asks whether those genes are expressed more in the same type (p-value from random loci) |
| `<grouping>_gene_sets_umap.png` | the genes of each pool's specific loci as a gene set, scored on the RNA UMAP; the AUROC says how specifically a set marks its own RNA type |
| `structure_metrics.csv`, `*.csv` | all scores, tracks, loci and gene sets |

`<grouping>` is `predicted` or `truth`. With `--wandb` only the metrics are logged, not the figures.

Options: `--structure_scool` (single cells at a finer resolution than the model, 100kb works well; default `--scool`), `--structure_markers` (TSV with `celltype` and `gene` columns; default: the dataset's markers in `evals.DATASET_CONFIGS`, if any, otherwise the marker plots are skipped), `--structure_min_cells` (smallest pool, 50), `--structure_max_cells` (pools are subsampled to this many cells, 1000, which bounds the run time to roughly 10-30 minutes), `--structure_top_loci` (15).

A pool of predicted cells whose structure matches the expression of its RNA type is evidence that the predictions are biologically meaningful, also for cells without labels.

## Figures during training

Every `--save_interval` epochs the label transfer metrics are logged (wandb) and figures are written to `<exp_name>/viz/{pretrain,finetune}/`:

| Folder | Content |
| --- | --- |
| `cells/` | Hi-C cells, UMAP and PCA: annotation, transferred labels, Leiden, depth, batch |
| `joint/` | all modalities in the joint space (subsampled to 30k cells each); paired cells vs random pairs |
| `confusion/` | annotated vs transferred labels, row normalized (plus sorted α/β cells) |
| `params/` | decoder parameters: strata weights by cell type, scales, zero inflation, dispersion, composition weights or stratum keys, attention range |
| `features/` | anchor and gene embeddings of one chromosome and the similarity of its anchor embeddings (overwritten each time) |
| `loci/` | for the first `--viz_loci` (3) genes of `--gene_list`: guidance graph arcs, observed vs predicted contacts per cell type, decoder attention (overwritten each time) |

The numbered figures (`cells`, `joint`, `confusion`, `params`) make animations of training. Attention maps reproduce the attention weights of the decoder exactly (checked against its output at every save). `--viz basic` skips features and loci, `--viz none` writes no figures.

## Baselines

`--baselines scgad` runs scGAD on the same cells after training (`scgad.py`, a Python port of `scGAD` from [BandNorm](https://github.com/keleslab/BandNorm)): contacts within each gene body are counted per cell, depth normalized and z-scored per gene, co-embedded with the RNA by PCA on shared variable genes and Harmony, and evaluated with the same label transfer and metrics as HiGLUE. Results go to `<eval_dir>/baselines/scgad/`; its `alone_*` metrics are the scGAD embedding without RNA.

```bash
python higlue.py ... --train --baselines scgad --scgad_scool hic_5kb.scool SCORE ...
```

scGAD is meant for high resolution: pass a 5-10kb `.scool` with `--scgad_scool` (default `--scool`). GAD counts are cached in `<data_dir>/baselines/`. Options: `--scgad_features` (shared variable genes, 2000), `--scgad_pcs` (30), `--scgad_jobs` (processes reading cells, 8). It also runs on its own: `python scgad.py --rna rna.h5ad --scool hic_5kb.scool --reference hic_ref.tsv --out_dir <dir>`.

`--baselines graph_match` (`graph_match.py`, single resolution) labels the Hi-C cells without any alignment or composition matching:

1. **Hi-C only clusters:** every stratum of a cell's band features is scaled to the same total, then log, PCA, Harmony by batch (when there are several batches) and Leiden. `--graph_match_resolution auto` (default) clusters at 0.5–3 and keeps the resolution whose clusters match their types best (size-weighted mean z of step 2): clusters that merge types match none of them well. On PFC it picks 2.0, on the islet atlas 0.5, in both the scan's best composition estimate.
2. **Names through the graph:** each cluster's same-bin pseudobulk contacts per anchor are correlated with every RNA type's expression signature. The signatures are carried to the anchors by the guidance graph's gene–anchor edges. The best type, as a z-score over the types, names the cluster.

It measures what the guidance graph alone can do. Results go to `<eval_dir>/baselines/graph_match/`:

- the HiGLUE metrics, plus `composition_tv_to_annotation` (its composition estimate vs the annotation) and `higlue_composition_tv_to_annotation` (HiGLUE's predictions vs the annotation)
- `cluster_matching.png`: clusters × RNA types
- `composition.png`: the annotation, graph matching, HiGLUE and the reference side by side
- `hic_umap.png` and `confusion.png`

Standalone on preprocessed data: `python graph_match.py --data_dir main_data/<dataset>/<config> --dset <name> --out_dir <dir>`. `islet_benchmark/graph_cluster_matching.py` runs it on every dataset.

`--baselines loops --loop_baseline_loops loops.tsv` (`loop_features.py`) embeds the Hi-C cells without RNA from their contacts at a given set of loops only (TSV with `chrom`, `pos1`, `pos2` in the coordinates of `--scgad_scool`/`--scool`), following the loop-based embedding of Tian et al., Science 2023 (Fig. 2H): each loop is a pixel at `--loop_baseline_res` (100kb), read from the high-resolution scool, with feature values from `--loop_baseline_norm`: `lsi` (the paper's binary LSI), `depth`, `oe` (per-cell distance expected), `ice` (per-cell ICE biases at 1 Mb) or `ice_oe`. It is scored on the labelled cells like the scGAD `alone_*` metrics plus the paper's K-Means benchmark (`kmeans_ari`: best ARI over k = 3-12), and repeated on the same pixels moved 1-5 Mb (`shifted_*`), so the gap shows what the loop positions add beyond distance-matched contacts. Results go to `<eval_dir>/baselines/loops/`. Standalone: `python loop_features.py --scool hic_5kb.scool --reference hic_ref.tsv --loops loops.tsv --out_dir <dir>`.

### Gene-body contacts as a modality

The scGAD gene-body contact counts of the Hi-C cells are added to HiGLUE as a third modality, `gad` (on by default for single-resolution models; `--no-gad_modality` turns it off). It has one count feature `gad:<gene>` per variable RNA gene with gene-body contacts in at least `--gad_min_cells` (1%) of the cells, modeled as negative binomial like expression. Each feature is linked to its gene in the guidance graph. Its cells are the Hi-C cells, so the model trains as a paired model (`PairedSCGLUEModel`): the Hi-C and gene-body encoders of a cell are tied by the cross-prediction and cosine losses.

The counts come from `--gad_scool` (default `--scgad_scool`, else `--scool`); finer resolutions separate more genes. They are cached with the scGAD baseline's. `--gad_weight` sets the modality weight (1.0, Hi-C uses `--hic_weight`). Metrics of the gene-body encoder are reported with a `gad_` prefix, and the Hi-C metrics stay unprefixed. Pass a high-resolution `.scool` (`--scgad_scool hic_5kb.scool`); without one, training warns and counts at the model resolution.

### Per-cell distance-decay normalization

By default the Hi-C input of a cell is normalized by its total contacts only. Its distance-decay profile (how its contacts are spread over strata) is kept, and the decoder predicts each cell's split over strata from its embedding (`--strata_weights cell`). Single-cell distance decay varies with cell cycle and state, but also with the library. In the islet atlas, one donor has a cis/trans ratio about 1.5× higher and more short-range contacts in every cell type.

`--decay_norm cell` removes the profile:

- **Encoder:** every stratum of a cell is scaled by the cell's own total in that stratum, so the input only says where along the genome the contacts at each distance fall.
- **Decoder:** each stratum is decoded with its observed total as library size, so the embedding no longer has to explain the profile.

It applies to single-resolution models only, and it is off by default until it is benchmarked. In the in-house islet Dip-C, the FACS-sorted plates have 72% of their contacts in the same bin (52–62% in the other plates) at half the depth. Without `--decay_norm cell`, 77–98% of the sorted cells sit in clusters of sorted cells only, and their labels depend on how those clusters happen to align (sorted accuracy 0.04–0.98 over seeds). With it, no cluster is sorted-only and seeds agree (0.86–0.89):

```bash
TRAIN_ARGS="--decay_norm cell" ./benchmark_main.sh --datasets islet_atlas,islet_inhouse
```

### Composition-weighted alignment

The modality discriminator asks the Hi-C and RNA embeddings to overlap. When the RNA reference has different cell-type proportions from the Hi-C cells, overlap is only possible by moving Hi-C cells onto the reference's over-represented types. In the islet atlas, acinar and ductal cells were moved onto Alpha, PSC and Endothelial. GLUE's `--balance` estimates its weights from the pretrained embedding, which is already misaligned.

`--comp_weight graph` (the default for unpaired data, see below) estimates the Hi-C composition first, without the discriminator, using the graph matching above. It then weights every RNA cell in the discriminator by (estimated Hi-C share / reference share) of its type. The reference then has the estimated Hi-C composition, as with `--match_composition` but without the Hi-C labels.

Details:
- The estimate counts each cluster's cells toward its name. `--comp_weight_temperature T` spreads them over the types by a softmax of the match z-scores instead; that hurt the islet atlas, where similar types (Beta/Gamma, acinar/ductal) have close z-scores.
- `--comp_weight_floor` (0.1) is the share of weight kept by reference types the estimate does not find.
- It helps when the Hi-C only clusters resolve the reference's types. Estimate vs annotation (total variation; reference vs annotation in brackets): islet atlas 0.03 (0.33), PFC 0.11 (0.13), human brain 0.36 (0.20). The human brain reference has 19 types, finer than the Hi-C clusters, and the estimate is worse than the reference itself. `<eval_dir>/comp_weight/summary.csv` records the share of the reference the named types cover (`reference_coverage`; atlas 0.93, PFC 0.90, human brain 0.83), and training warns below 0.85.
- `--comp_weight auto` (default) is `graph` for single-resolution models of unpaired data, and `none` when most Hi-C cells are also RNA cells (identical composition by construction), with `--resolutions`, `--balance` or `--match_composition`.
- Islet atlas accuracy 0.83–0.91 with it vs 0.66–0.73 without (GAD modality, 3 seeds); PFC 0.817 vs 0.828 (its reference already matches).
- The estimate, the weights and a composition figure go to `<eval_dir>/comp_weight/`.
- With `--baselines graph_match`, the same clusters are reused.

```bash
TRAIN_ARGS="--comp_weight none" ./benchmark_main.sh --datasets islet_atlas   # ablation
```

## Main benchmark

The inputs of the paper's main results are collected in `datasets/`, one directory per dataset with the same file names (`rna.h5ad`, `genes.gtf`, `hic_ref.tsv`, `hic_<res>.scool`):

| Dataset | scHi-C | RNA reference | Genome |
| --- | --- | --- | --- |
| `pfc` | sn-m3C-seq PFC neurons (Lee et al. 2019) | snRNA-seq (Velmeshev et al. 2019) | hg19 |
| `human_brain` | snm3C-seq brain atlas neurons (Tian et al. 2023) | snRNA-seq atlas (Siletti et al. 2023) | hg38 |
| `hires_brain` | HiRES mouse brain, paired (Liu et al. 2023) | same cells | mm10 |
| `droplet_cortex` | Droplet Hi-C mouse cortex, paired multiome (Chang et al. 2025) | same cells | mm10 |
| `islet_atlas` | all cells of the islet samples of the snm3C-seq human body atlas (Zhou et al. 2026), incl. stromal and immune cells (`datasets/build_islet_atlas_all.py`) | PancDB / HPAP | hg38 |
| `islet_inhouse` | in-house islet Dip-C (sorted α/β labeled) | PancDB / HPAP | hg19 |

```bash
datasets/link_datasets.sh          # symbolic links to the lab storage (source paths are listed in this script)
datasets/sync_to_cluster.sh        # copy the tree to the cluster (links within it are kept, ~30 GB)

./benchmark_main.sh --list         # per dataset settings
./benchmark_main.sh --prep_only    # preprocess every dataset once
./benchmark_main.sh                # 3 seeds per dataset: train, evaluate, scGAD, graph matching, structure (islets)
python benchmark_summary.py        # mean ± SD tables (main_runs/main_summary.csv) and main_comparison.png
```

On SLURM, `benchmark_main.slurm` runs the preprocessing, then an array of training shards (see the header of the file). `benchmark_final.sh` submits the final benchmark: one preprocessing job, then one array per configuration (the defaults, `--decay_norm cell`, and the ablations `--comp_weight none` and `--no-gad_modality --comp_weight none`), one run per task. Runs go to `main_runs/<dataset>/<res>_<config>_gadcw[-train-<hash>]/seed<N>/` (`gadcw`: the model defaults, `MODEL_TAG`), preprocessed data to `main_data/`. `EXTRA_ARGS="..."` adds higlue.py options to every step (own preprocessing), `TRAIN_ARGS="..."` to training only. `benchmark.sh` and `islet_benchmark/sweep.sh` hold the ablations.

## Multi-resolution models (experimental)

`--resolutions 500kb 100kb ...` trains one model on Hi-C features of several resolutions at once, built from the finest `.scool`. This mode is under active development; see `python higlue.py --help SCORE` (options starting with `--multires_`).
