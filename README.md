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

## Multi-resolution models (experimental)

`--resolutions 500kb 100kb ...` trains one model on Hi-C features of several resolutions at once, built from the finest `.scool`. This mode is under active development; see `python higlue.py --help SCORE` (options starting with `--multires_`).
