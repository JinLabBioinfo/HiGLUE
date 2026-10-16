import os
import re
import anndata as ad
import networkx as nx
import scanpy as sc
import scglue
import sys
import argparse
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import torch

from matplotlib.colors import Normalize
from preprocess_data import (
    preprocess_higlue, preprocess_higlue_multires, multires_extra_suffix
)
import multires_hic as mh
import evals
import structure_eval


def _sanitize_plot_name(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_") or "gene"


def _resolve_gene_indices(var_names, gene_list):
    var_name_lookup = {}
    for var_name in var_names:
        var_name_lookup.setdefault(str(var_name).lower(), []).append(var_name)

    resolved = []
    missing = []
    ambiguous = []
    for gene_name in gene_list or []:
        if gene_name in var_names:
            resolved_name = gene_name
        else:
            matches = var_name_lookup.get(str(gene_name).lower(), [])
            if len(matches) == 1:
                resolved_name = matches[0]
            elif len(matches) > 1:
                ambiguous.append((gene_name, matches))
                continue
            else:
                missing.append(gene_name)
                continue
        resolved.append((gene_name, resolved_name, int(var_names.get_loc(resolved_name))))
    return resolved, missing, ambiguous


@torch.no_grad()
def _decode_selected_rna_features(glue_model, source_key, source_adata, graph, gene_indices, batch_size=128):
    if not gene_indices:
        return np.empty((source_adata.shape[0], 0), dtype=np.float32)

    net = glue_model.net
    device = net.device
    net.eval()

    latent = glue_model.encode_data(source_key, source_adata, batch_size=batch_size)
    feature_embeddings = glue_model.encode_graph(graph)
    rna_feature_index = getattr(net, "rna_idx")
    if torch.is_tensor(rna_feature_index):
        rna_feature_index = rna_feature_index.detach().cpu().numpy()
    else:
        rna_feature_index = np.asarray(rna_feature_index)
    feature_embeddings = feature_embeddings[rna_feature_index]
    feature_embeddings = torch.as_tensor(feature_embeddings, dtype=torch.float32, device=device)
    decoder = net.u2x["rna"]

    target_batch_key = glue_model.modalities["rna"]["use_batch"]
    target_batches = glue_model.modalities["rna"]["batches"]
    if target_batch_key and target_batch_key in source_adata.obs:
        batch_codes = target_batches.get_indexer(source_adata.obs[target_batch_key])
        batch_codes = np.where(batch_codes < 0, 0, batch_codes)
    else:
        batch_codes = np.zeros(source_adata.shape[0], dtype=int)

    decoded = []
    for start in range(0, latent.shape[0], batch_size):
        stop = min(start + batch_size, latent.shape[0])
        latent_batch = torch.as_tensor(latent[start:stop], dtype=torch.float32, device=device)
        batch_codes_batch = torch.as_tensor(batch_codes[start:stop], dtype=torch.int64, device=device)
        target_libsize = torch.ones((stop - start, 1), dtype=torch.float32, device=device)
        decoded.append(
            decoder(latent_batch, feature_embeddings, batch_codes_batch, target_libsize)
            .mean[:, gene_indices]
            .detach()
            .cpu()
            .numpy()
        )
    return np.vstack(decoded)


def _plot_overlay_umap(adata, values, title, output_path):
    coords = adata.obsm.get("X_umap")
    if coords is None:
        raise ValueError("UMAP coordinates not found in AnnData object.")

    values = np.asarray(values, dtype=float)
    finite_mask = np.isfinite(values)
    if not finite_mask.any():
        print(f"Skipping {title}: no finite values available.")
        return False

    plot_coords = coords[finite_mask]
    plot_values = values[finite_mask]
    order = np.argsort(plot_values)
    plot_coords = plot_coords[order]
    plot_values = plot_values[order]

    if np.allclose(plot_values, plot_values[0]):
        vmin = float(plot_values[0])
        vmax = float(plot_values[0] + 1e-6)
    else:
        vmin, vmax = np.nanpercentile(plot_values, [1, 99])
        if vmax <= vmin:
            vmin = float(np.nanmin(plot_values))
            vmax = float(np.nanmax(plot_values))
        if vmax <= vmin:
            vmax = vmin + 1e-6

    norm = Normalize(vmin=vmin, vmax=vmax)
    normalized_values = np.clip(norm(plot_values), 0.0, 1.0)
    colors = plt.cm.magma(normalized_values)
    colors[:, 3] = 0.08 + 0.92 * normalized_values

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(coords[:, 0], coords[:, 1], s=10, c="lightgray", alpha=0.3, linewidths=0, rasterized=True)
    scatter = ax.scatter(
        plot_coords[:, 0], plot_coords[:, 1], s=12, c=colors, linewidths=0, rasterized=True
    )
    scatter.set_clim(vmin, vmax)
    ax.set_title(title)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

    scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap="magma")
    scalar_mappable.set_array([])
    colorbar = fig.colorbar(scalar_mappable, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Predicted transcriptional activity")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True

if __name__ == '__main__':
    glue_parser = argparse.ArgumentParser()
    # preprocessing args
    glue_parser.add_argument('--seed', type=int, default=25)
    glue_parser.add_argument('--data_dir', type=str, default='')
    glue_parser.add_argument('--loop_q', type=str, default='0.98')
    glue_parser.add_argument('--n_strata', type=int, default=10)
    glue_parser.add_argument('--use_ice', action='store_true')
    glue_parser.add_argument('--use_dist_norm', action='store_true')
    glue_parser.add_argument('--viz_rna', action='store_true')
    glue_parser.add_argument('--load_rna', action='store_true')
    glue_parser.add_argument('--preprocess', action='store_true')
    glue_parser.add_argument('--train', action='store_true')
    glue_parser.add_argument('--offset', type=int, default=0)
    glue_parser.add_argument('--min_count', type=int, default=0)
    glue_parser.add_argument('--distal_interactions', type=int, default=None)
    glue_parser.add_argument('--filter_strata', type=float, default=None)
    glue_parser.add_argument('--exclusive_strata', action='store_true')
    glue_parser.add_argument('--use_xy', action='store_true')
    glue_parser.add_argument('--n_genes', type=int, default=10000)
    glue_parser.add_argument('--n_atac_peaks', type=int, default=None)
    glue_parser.add_argument('--gene_list', nargs='+', default=None)
    glue_parser.add_argument(
        '--match_composition', action='store_true',
        help='diagnostic oracle: downsample the RNA reference to the cell type '
             'proportions of the Hi-C query (uses the Hi-C labels)'
    )
    glue_parser.add_argument(
        '--celltypes', nargs='+', default=None,
        help='train and evaluate on cells of these types only, in every modality. '
             'Query (Hi-C) names also select the matching RNA subtypes through '
             'the dataset label map in evals.py (e.g. "Epi Aci" keeps RNA Acinar)'
    )
    glue_parser.add_argument('--no_depth_correction', action='store_true')
    glue_parser.add_argument('--use_trans', action='store_true')
    glue_parser.add_argument('--bulk_hic', type=str, default=None)
    glue_parser.add_argument('--coassay', type=str, nargs='+', default=None)

    # SCORE args
    glue_parser.add_argument('--rna_file', type=str, default=None)
    glue_parser.add_argument('--atac_file', type=str, default=None)
    glue_parser.add_argument('--methyl_file', type=str, default=None)
    glue_parser.add_argument('--gtf', type=str, default=None)
    glue_parser.add_argument('--dset', type=str, default=None)
    glue_parser.add_argument('--subname', type=str, default=None)
    glue_parser.add_argument('--scool', type=str, default=None)
    glue_parser.add_argument('--reference', type=str, default=None)
    glue_parser.add_argument('--resolution', type=str, default='100kb')
    glue_parser.add_argument('--min_depth', type=int, default=40000)
    glue_parser.add_argument('--max_depth', type=int, default=1000000)
    glue_parser.add_argument('--ignore_chr_filter', action='store_true')
    glue_parser.add_argument('--ignore_filter', action='store_true')

    # extra training args
    glue_parser.add_argument('--prior', type=str, default='dcq')
    glue_parser.add_argument('--hic_weight', type=float, default=10.0)
    glue_parser.add_argument('--n_neighbors', type=int, default=15)
    glue_parser.add_argument('--lam_align', type=float, default=0.01)
    glue_parser.add_argument('--lam_graph', type=float, default=0.1)
    glue_parser.add_argument('--lam_cycle', type=float, default=0.02)
    glue_parser.add_argument('--suffix', type=str, default='2d')
    glue_parser.add_argument('--exp_name', type=str, default=None)
    glue_parser.add_argument('--save_interval', type=int, default=100)
    glue_parser.add_argument('--latent_dim', type=int, default=64)
    glue_parser.add_argument('--batch_size', type=int, default=128)
    glue_parser.add_argument('--h_dim', type=int, default=128)
    glue_parser.add_argument('--h_depth', type=int, default=2)
    glue_parser.add_argument('--neg_samples', type=int, default=10)
    glue_parser.add_argument('--wait_n_lrs', type=int, default=2)
    glue_parser.add_argument('--lr', type=float, default=2e-3)
    glue_parser.add_argument('--max_epochs', type=int, default=None)
    glue_parser.add_argument('--wandb', action='store_true')
    glue_parser.add_argument('--balance', action='store_true')
    glue_parser.add_argument('--normalize_u', action='store_true')
    glue_parser.add_argument('--binarize', action='store_true')
    glue_parser.add_argument('--use_batch', type=str, default=None)
    glue_parser.add_argument('--use_rna_pca', action='store_true')
    glue_parser.add_argument('--use_atac_counts', action='store_true')
    glue_parser.add_argument('--cache_checkpoint', type=str, default=None)
    glue_parser.add_argument(
        '--eval_dir', type=str, default=None,
        help='where evaluation metrics, plots and embeddings are written '
             '(default: <exp_name>/eval)'
    )
    # chromatin structure analysis at the end of training (structure_eval.py)
    glue_parser.add_argument(
        '--structure_eval', action='store_true',
        help='after training, pool Hi-C cells by predicted and annotated type into '
             'pseudobulk coolers and analyze insulation and A/B compartments at marker '
             'genes and at cell type specific loci, next to the RNA expression of their '
             'genes (writes <eval_dir>/structure)'
    )
    glue_parser.add_argument('--structure_scool', type=str, default=None,
                             help='single-cell coolers for the structure analysis (default: --scool; '
                                  'a finer resolution than the model, e.g. 100kb, works best)')
    glue_parser.add_argument('--structure_binsize', type=int, default=None,
                             help='pseudobulk grid in bp (default: the scool resolution, at least 100kb)')
    glue_parser.add_argument('--structure_markers', type=str, default=None,
                             help='TSV/CSV with columns celltype, gene (default: the dataset markers in evals.py)')
    glue_parser.add_argument('--structure_min_cells', type=int, default=50, help='smallest pool')
    glue_parser.add_argument('--structure_max_cells', type=int, default=1000,
                             help='pools are subsampled to at most this many cells')
    glue_parser.add_argument('--structure_top_loci', type=int, default=15,
                             help='cell type specific loci per pool and feature')
    glue_parser.add_argument(
        '--strata_weights', choices=['cell', 'global'], default='cell',
        help="how the library size is split across strata: 'cell' predicts it "
             "from the cell embedding (lets the model represent per-cell "
             "distance decay), 'global' shares one profile across all cells"
    )
    glue_parser.add_argument(
        '--no_strata_input_norm', action='store_true',
        help='do not divide each stratum by its typical magnitude before the '
             'log transform of the encoder input'
    )
    glue_parser.add_argument(
        '--lam_downsample', type=float, default=0.0,
        help='weight of the depth consistency loss, which pulls a cell and a '
             'downsampled copy of it to the same embedding (0 disables it)'
    )
    glue_parser.add_argument('--downsample_min', type=float, default=0.3)
    glue_parser.add_argument('--downsample_max', type=float, default=0.9)

    # multi-resolution args
    glue_parser.add_argument(
        '--resolutions', nargs='+', default=None,
        help='train a single multi-resolution model on these resolutions '
             '(e.g. 500kb 100kb 5kb). Coarser grids are derived from the '
             '.scool passed to SCORE, which must be the finest resolution.'
    )
    glue_parser.add_argument(
        '--multires_strata', nargs='+', type=int, default=None,
        help='number of diagonal strata per resolution (one value, or one per '
             'resolution); defaults to --n_strata'
    )
    glue_parser.add_argument(
        '--multires_max_anchors', nargs='+', type=int, default=None,
        help='maximum number of anchors kept per resolution (0 = keep all '
             'detected anchors); defaults to 50000'
    )
    glue_parser.add_argument(
        '--multires_tile_size', type=int, default=8,
        help='anchors are kept in contiguous tiles of this many bins'
    )
    glue_parser.add_argument(
        '--multires_stat_cells', type=int, default=None,
        help='number of cells scanned to rank anchors and contacts '
             '(default: all cells)'
    )
    glue_parser.add_argument(
        '--multires_min_frac', type=float, default=0.01,
        help='an anchor must be detected in at least this fraction of cells'
    )
    glue_parser.add_argument('--multires_chunk_size', type=int, default=256)
    glue_parser.add_argument('--multires_use_dist', action='store_true',
                             help='also add distance-decay gene-anchor edges')
    glue_parser.add_argument('--multires_dist_window', type=int, default=150000)
    glue_parser.add_argument(
        '--multires_res_dim', type=int, default=None,
        help='dimensionality of each per-resolution cell embedding '
             '(default: --h_dim)'
    )
    glue_parser.add_argument(
        '--multires_mlp_max_band', type=int, default=262144,
        help='resolutions whose band matrix is larger than this are encoded '
             'with convolutions instead of a dense projection'
    )
    glue_parser.add_argument(
        '--multires_anchor_subsample', type=int, default=None,
        help='reconstruct only this many anchors per resolution in each '
             'training step (keeps high resolution training tractable)'
    )
    glue_parser.add_argument('--multires_conv_channels', nargs=3, type=int,
                             default=(8, 16, 32))
    glue_parser.add_argument('--multires_conv_patch', nargs=2, type=int,
                             default=(2, 8))
    glue_parser.add_argument('--multires_conv_pool_width', type=int, default=32)
    glue_parser.add_argument(
        '--multires_use_attn', action=argparse.BooleanOptionalAction, default=True,
        help='use local attention over anchor embeddings in the '
             'multi-resolution decoder (on by default)'
    )
    glue_parser.add_argument(
        '--multires_attn_window', type=int, default=None,
        help='local attention window in anchors; defaults to 4x the number of '
             'strata of each resolution (clamped to 32-256)'
    )
    glue_parser.add_argument(
        '--multires_checkpoint', action='store_true',
        help='recompute the per-resolution encoders during the backward pass '
             'to trade compute for GPU memory'
    )
    glue_parser.add_argument(
        '--backed', action='store_true',
        help='read the Hi-C dataset from disk one minibatch at a time'
    )

    glue_args = sys.argv.index('SCORE')
    args = glue_parser.parse_args(sys.argv[1:glue_args] + sys.argv[glue_args + 1:])

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_name = args.dset
    out_dir = f'data/{dataset_name}_data'
    if args.data_dir != '':
        out_dir = f'{args.data_dir}'
    os.makedirs(out_dir, exist_ok=True)
    prior_name = args.prior
    resolution = args.resolution
    n_distal_interactions = args.distal_interactions
    filter_strata = args.filter_strata
    exclusive_strata = args.exclusive_strata
    use_xy = args.use_xy
    seed = args.seed
    min_count = args.min_count
    n_genes = args.n_genes
    hic_type = 'raw'
    if args.use_ice:
        hic_type = 'ice'
    loop_q = args.loop_q
    use_trans = args.use_trans
    depth_correction = not args.no_depth_correction
    hic_weight = args.hic_weight
    n_neighbors = args.n_neighbors
    lam_align = args.lam_align
    lam_graph = args.lam_graph
    lam_cycle = args.lam_cycle
    suffix = args.suffix
    coassay = args.coassay
    if coassay is None:
        coassay = []
    exp_name = args.exp_name
    save_interval = args.save_interval
    latent_dim = args.latent_dim
    batch_size = args.batch_size
    h_dim = args.h_dim
    h_depth = args.h_depth
    n_strata = args.n_strata
    neg_samples = args.neg_samples
    wait_n_lrs = args.wait_n_lrs
    lr = args.lr
    max_epochs = args.max_epochs if args.max_epochs is not None else "AUTO"
    normalize_u = args.normalize_u
    full_file_suffix = f"{resolution}_{hic_type}_{loop_q}_{suffix}_{n_strata}"
    graph_file_suffix = f"{prior_name}_prior_{resolution}_{hic_type}_{loop_q}_{suffix}_{n_strata}"
    use_rep = None
    min_depth = args.min_depth
    use_attn = True
    binarize = args.binarize
    use_wandb = args.wandb
    skip_balance = not args.balance
    use_rna_pca = args.use_rna_pca
    use_batch = args.use_batch
    cache_checkpoint = args.cache_checkpoint
    atac_file = args.atac_file
    methyl_file = args.methyl_file

    multires = bool(args.resolutions)
    if multires:
        # Resolutions are always ordered coarse to fine
        resolutions = sorted(args.resolutions, key=mh.parse_resolution, reverse=True)
        multires_strata = mh.broadcast_option(
            args.multires_strata, len(resolutions), "multires_strata",
            default=n_strata
        )
        multires_strata = [int(s) for s in multires_strata]
        full_file_suffix = mh.multires_suffix(
            resolutions, multires_strata, float(loop_q),
            extra=multires_extra_suffix(args.use_dist_norm, args.use_trans)
        )
        graph_file_suffix = f"{prior_name}_prior_{full_file_suffix}"
        print(f"Multi-resolution mode: {list(zip(resolutions, multires_strata))}")

    if args.preprocess:
        if multires:
            preprocess_higlue_multires(args, glue_args)
        else:
            preprocess_higlue(args, glue_args)

    if args.train:
        prior = nx.read_graphml(f"{out_dir}/graphs/{graph_file_suffix}.graphml.gz")
        rna = ad.read_h5ad(f"{out_dir}/rna/rna_{full_file_suffix}.h5ad")
        if atac_file is not None:
            atac = ad.read_h5ad(f"{out_dir}/atac/atac_{full_file_suffix}.h5ad")
        if methyl_file is not None:
            methyl = ad.read_h5ad(f"{out_dir}/methyl/methyl_{full_file_suffix}.h5ad")
        hic = ad.read_h5ad(
            f"{out_dir}/hic/hic_{full_file_suffix}.h5ad",
            backed="r" if args.backed else None
        )
        if args.backed:
            print(f"Reading Hi-C data lazily from disk: {hic.shape}")
        try:
            hic.obs.loc[
                hic.obs_names.str.contains('alpha', case=False, regex=False, na=False), 'celltype'
            ] = 'Alpha'
        except Exception as e:
            print(e)
        try:
            hic.obs.loc[
                hic.obs_names.str.contains('beta', case=False, regex=False, na=False), 'celltype'
            ] = 'Beta'
        except Exception as e:
            print(e)

        rna.var["highly_variable"] = rna.var["highly_variable"] & rna.var["in_hic"]
        hic.var["highly_variable"] = True
        if args.backed:
            # Subsetting a backed dataset would pull it into memory, so cells
            # have to be filtered while preprocessing (SCORE's --min_depth)
            n_shallow = int((hic.obs['depth'] <= min_depth).sum())
            if n_shallow:
                print(f"WARNING: keeping {n_shallow} cells with depth <= {min_depth}; "
                      f"pass --min_depth to SCORE to drop them during preprocessing.")
        else:
            hic = hic[hic.obs['depth'] > min_depth, :]
        hic.obs['read_depth'] = hic.obs['depth'].copy()  # for visualization later

        # set depth as fraction of total counts
        rna.obs['depth'] = rna.layers['counts'].sum(axis=1)
        rna.obs['depth'] = rna.obs['depth'] / rna.obs['depth'].max()
        if atac_file is not None:
            atac.obs['depth'] = atac.layers['counts'].sum(axis=1)
            atac.obs['depth'] = atac.obs['depth'] / atac.obs['depth'].max()
        if methyl_file is not None:
            methyl.obs['depth'] = methyl.layers['counts'].sum(axis=1)
            methyl.obs['depth'] = methyl.obs['depth'] / methyl.obs['depth'].max()
        # set hic depth per batch
        for batch in hic.obs['batch'].unique():
            mask = hic.obs['batch'] == batch
            if multires:
                # Multi-resolution data is streamed from disk, so the sequencing
                # depth recorded during preprocessing is used directly
                batch_depth = hic.obs.loc[mask, 'read_depth'].astype(float)
                hic.obs.loc[mask, 'depth'] = (batch_depth / batch_depth.max()).values
                continue
            batch_hic = hic[mask, :].copy()
            batch_hic.obs['depth'] = batch_hic.layers['counts'].sum(axis=1)
            batch_hic.obs['depth'] = batch_hic.obs['depth'] / batch_hic.obs['depth'].max()
            hic.obs.loc[mask, 'depth'] = batch_hic.obs['depth'].values

        # set hic depth
        # hic.obs['depth'] = hic.layers['counts'].sum(axis=1)
        # hic.obs['depth'] = hic.obs['depth'] / hic.obs['depth'].max()


        if binarize and not multires:
            hic.X = np.int32(hic.X > 0)
            hic.layers['counts_pre_binarize'] = hic.layers['counts'].copy()
            hic.layers['counts'] = hic.X.copy()

        celltypes = list(hic.obs['celltype'].unique())
        if 'pfc' in dataset_name:
            neurons_rna_celltypes = ['IN-PV', 'IN-SST', 'IN-VIP', 'IN-SV2C', 'Neu-mat',
                      'L2/3', 'L4', 'L5', 'L5/6', 'L6', 'L5/6-CC']
            neurons_hic_celltypes = ['L2/3', 'L4', 'L5', 'L6', 'Ndnf', 'Pvalb', 'Sst', 'Vip']  

            rna_celltype_map = {'AST-FB': 'Astro',
                                'AST-PP': 'Astro',
                                'Endothelial': 'Endo',
                                'IN-PV': 'Pvalb',
                                'IN-SST': 'Sst',
                                'IN-VIP': 'Vip',
                                'IN-SV2C': 'Ndnf',
                                'L2/3': 'L2/3',
                                'L4': 'L4',
                                'L5': 'L5',
                                'L6': 'L6',
                                'L5/6': 'L6',
                                'L5/6-CC': 'L5',
                                'Microglia': 'MG',
                                'Neu-NRGN-I': 'Neu',
                                'Neu-NRGN-II': 'Neu',
                                'Neu-mat': 'Neu',
                                'OPC': 'OPC',
                                'Oligodendrocyte': 'ODC'}
            rna = rna[rna.obs['region'] == 'PFC', :]
            rna = rna[~rna.obs['celltype'].isin(['Neu-NRGN-I', 'Neu-NRGN-II', 'Neu-mat'])]
            #rna = rna[~rna.obs['celltype'].isin(['Neu-NRGN-I', 'Neu-NRGN-II'])]
            rna = rna[rna.obs['celltype'].isin(neurons_rna_celltypes)]
            rna.obs['celltype'] = rna.obs['celltype'].map(rna_celltype_map)
        elif 'human_brain' in dataset_name:
            neurons_rna_celltypes = ['IN-PV', 'IN-SST', 'IN-VIP', 'IN-SV2C', 'Neu-mat',
                      'L2/3', 'L4', 'L5', 'L5/6', 'L6', 'L5/6-CC']
            neurons_hic_celltypes = ['L2/3', 'L4', 'L5', 'L6', 'Ndnf', 'Pvalb', 'Sst', 'Vip']  

            rna_celltype_map = {'Amygdala excitatory': 'Amy-Exc',
                                'Deep-layer corticothalamic and 6b': 'L6b',
                                'Deep-layer intratelencephalic': 'L2/3-IT',
                                'Upper-layer intratelencephalic': 'L2/3-IT',
                                'Deep-layer near-projecting': 'L5/6-NP',
                                'Eccentric medium spiny neuron': 'MSN-D1',
                                'Medium spiny neuron': 'MSN-D1',
                                'LAMP5-LHX6 and Chandelier': 'Lamp5-Lhx6',
                                'INT-LAMP5': 'Lamp5',
                                'L4 IT': 'L4-IT',
                                'L2/3 IT': 'L2/3-IT',
                                'L5 IT': 'L5-IT',
                                'L6 IT Car3': 'L6-IT-Car3',
                                'INT-VIP': 'Vip',
                                'INT-PVALB': 'Pvalb',
                                'INT-SST': 'Sst',
                                'L6 IT': 'L6-IT',
                                'CHAND': 'Pvalb-ChC',
                                'INT-SST-CHODL': 'Sst',
                                'INT-LAMP5-LHX6': 'Lamp5-Lhx6',
                                'L6 CT': 'L6-CT',
                                'L5/6 NP': 'L5/6-NP'
                                }
            remove_rna_celltypes = ['CB-GRAN', 'CB-MolLayerInt1', 'CB-MolLayerInt2', 'CGE interneuron', 'CB-PURK',
                                    'CB-PurkLayerInt', 'CB-GranLayerInt', 'UBC', 'Pax6',
                                    'DG-GRAN', 'Hippocampal CA1-3', 'Hippocampal CA4',
                                    'Upper rhombic lip', 'Midbrain-derived inhibitory', 'Thalamic excitatory', 'Miscellaneous',
                                    'Lower rhombic lip', 'Mammillary body', 'Splatter']

            hic_celltype_map = {}
            for celltype in rna.obs['celltype'].unique():
                if celltype not in rna_celltype_map:
                    rna_celltype_map[celltype] = celltype
            for celltype in hic.obs['celltype'].unique():
                if celltype not in hic_celltype_map:
                    hic_celltype_map[celltype] = celltype
            rna.obs['celltype'] = rna.obs['celltype'].map(rna_celltype_map)

        if args.celltypes:
            # after any dataset specific relabeling above, so the names given
            # match the final labels (preprocessing still used all cells)
            filter_modalities = {"rna": rna, "hic": hic}
            if atac_file is not None:
                filter_modalities["atac"] = atac
            if methyl_file is not None:
                filter_modalities["methyl"] = methyl
            filter_modalities = evals.filter_celltypes(
                filter_modalities, args.celltypes, label_key="celltype", dataset_name=dataset_name
            )
            rna, hic = filter_modalities["rna"], filter_modalities["hic"]
            if atac_file is not None:
                atac = filter_modalities["atac"]
            if methyl_file is not None:
                methyl = filter_modalities["methyl"]
            celltypes = list(hic.obs['celltype'].astype(str).unique())

        if args.match_composition:
            rna = evals.match_composition(rna, hic, label_key="celltype", dataset_name=dataset_name, seed=seed)

        rna_celltypes = rna.obs['celltype'].unique()
        rna.obs['celltype'] = pd.Categorical(rna.obs['celltype'], categories=rna_celltypes, ordered=True)
        hic.obs['celltype'] = pd.Categorical(hic.obs['celltype'], categories=celltypes, ordered=True)
        if atac_file is not None:
            atac_celltypes = atac.obs['celltype'].unique()
            atac.obs['celltype'] = pd.Categorical(atac.obs['celltype'], categories=atac_celltypes, ordered=True)
        if methyl_file is not None:
            methyl_celltypes = methyl.obs['celltype'].unique()
            methyl.obs['celltype'] = pd.Categorical(methyl.obs['celltype'], categories=methyl_celltypes, ordered=True)

        if use_wandb:
            import wandb
            wandb.init(project=f'GLUE-{dataset_name}-hic-2d-prior', 
                    sync_tensorboard=True, 
                    config={'prior': prior_name, 
                            'rna_file': args.rna_file,
                            'atac_file': args.atac_file,
                            'methyl_file': args.methyl_file,
                            'res': resolution,
                            'hic_type': hic_type,
                            'loop_q': loop_q,
                            'suffix': suffix, 
                            'min_depth': min_depth,
                            'n_distal_interactions': n_distal_interactions,
                            'filter_strata': filter_strata,
                            'min_count': min_count,
                            'n_genes': n_genes,
                            'celltypes': args.celltypes,
                            'match_composition': args.match_composition,
                            'rna_pca': use_rna_pca,
                            'latent_dim': latent_dim,
                            'h_dim': h_dim,
                            'h_depth': h_depth,
                            'n_strata': n_strata,
                            'batch_size': batch_size,
                            'neg_samples': neg_samples,
                            'use_trans': use_trans,
                            'normalize_u': normalize_u,
                            'use_attn': args.multires_use_attn if multires else use_attn,
                            'distances_normalized': args.use_dist_norm,
                            'binarize': binarize,
                            'lr': lr,
                            'hic_weight': hic_weight,
                            'n_neighbors': n_neighbors,
                            'lam_align': lam_align,
                            'lam_graph': lam_graph,
                            'lam_cycle': lam_cycle,
                            'lam_downsample': args.lam_downsample,
                            'strata_weights': args.strata_weights,
                            'strata_input_norm': not args.no_strata_input_norm,
                            'depth_correction': depth_correction,
                            'use_batch': use_batch,
                            'seed': seed,
                            'resolutions': resolutions if multires else [resolution],
                            'multires_strata': multires_strata if multires else [n_strata],
                            'multires_max_anchors': args.multires_max_anchors,
                            'multires_use_dist': args.multires_use_dist,
                            'multires_anchor_subsample': args.multires_anchor_subsample,
                            'exp_name': exp_name})

        scglue.models.configure_dataset(rna, "NB", use_highly_variable=True, use_layer="counts", use_rep="X_pca" if use_rna_pca else None,
                                        use_cell_type=None, use_batch=use_batch, use_depth="depth" if depth_correction else None,
                                        use_obs_names=True if 'rna' in coassay else False)
        if multires:
            scglue.models.configure_dataset(hic, "MultiResHiCZINB", use_highly_variable=False,
                                            use_layer=None, use_rep=None,
                                            use_depth="depth" if depth_correction else None, use_batch="batch",
                                            use_obs_names=True if 'hic' in coassay else False,
                                            use_multires=True, multires_res_order=resolutions,
                                            binarize=binarize)
        else:
            scglue.models.configure_dataset(hic, "HiCZINB", use_highly_variable=True, use_layer="counts", use_rep=None,
                                            use_depth="depth" if depth_correction else None, use_batch="batch",
                                            use_obs_names=True if 'hic' in coassay else False)
        if atac_file is not None:
            scglue.models.configure_dataset(atac, "NB", use_highly_variable=True, use_layer="counts", 
                                        use_cell_type=None, use_batch=use_batch, use_depth="depth" if depth_correction else None,
                                        use_obs_names=True if 'atac' in coassay else False)
        if methyl_file is not None:
            scglue.models.configure_dataset(methyl, "ZILN", use_highly_variable=True, use_layer="counts", 
                                        use_cell_type=None, use_batch=use_batch, use_depth="depth" if depth_correction else None,
                                        use_obs_names=True if 'methyl' in coassay else False)
        print(f"Total nodes in prior: {len(prior.nodes)}")

        print(rna.obs_names)
        print(hic.obs_names)
        rna.obs_names_make_unique()
        hic.obs_names_make_unique()
        if coassay is not None:
            # get the number of paired cells
            paired_obs = set()
            for assay in coassay:
                if assay == 'rna':
                    paired_obs = paired_obs.union(set(rna.obs_names).intersection(set(hic.obs_names)))
                elif assay == 'hic':
                    paired_obs = paired_obs.union(set(hic.obs_names).intersection(set(rna.obs_names)))
                elif assay == 'atac' and atac_file is not None:
                    paired_obs = paired_obs.union(set(atac.obs_names).intersection(set(rna.obs_names)))
                elif assay == 'methyl' and methyl_file is not None:
                    paired_obs = paired_obs.union(set(methyl.obs_names).intersection(set(rna.obs_names)))
            print(f"Number of paired cells: {len(paired_obs)}")
        if atac_file is not None:
            atac.obs_names_make_unique()
        if methyl_file is not None:
            methyl.obs_names_make_unique()

        if methyl_file is not None and atac_file is not None:
            dataset_dict = {"rna": rna, "hic": hic, "atac": atac, "methyl": methyl}
            modality_weights = {'rna': 1.0, 'hic': hic_weight, 'atac': 1.0, 'methyl': 1.0}
        elif methyl_file is not None:
            dataset_dict = {"rna": rna, "hic": hic, "methyl": methyl}
            modality_weights = {'rna': 1.0, 'hic': hic_weight, 'methyl': 1.0}
        elif atac_file is not None:
            dataset_dict = {"rna": rna, "hic": hic, "atac": atac}
            modality_weights = {'rna': 1.0, 'hic': hic_weight, 'atac': 1.0}
        else:
            dataset_dict = {"rna": rna, "hic": hic}
            modality_weights = {'rna': 1.0, 'hic': hic_weight}
        init_kws = {"latent_dim": latent_dim,
                    "use_attn": use_attn,
                    "binarize": binarize,
                    "h_dim": h_dim,
                    "h_depth": h_depth,
                    "n_strata": n_strata,
                    "cell_strata_weights": args.strata_weights == 'cell',
                    "strata_input_norm": not args.no_strata_input_norm,
                    "random_seed": seed}
        if multires:
            init_kws.update({
                "use_attn": args.multires_use_attn,
                "multires_use_attn": args.multires_use_attn,
                "multires_res_dim": args.multires_res_dim,
                "multires_mlp_max_band": args.multires_mlp_max_band,
                "multires_conv_channels": tuple(args.multires_conv_channels),
                "multires_conv_patch": tuple(args.multires_conv_patch),
                "multires_conv_pool_width": args.multires_conv_pool_width,
                "multires_anchor_subsample": args.multires_anchor_subsample,
                "multires_attn_window": args.multires_attn_window,
                "multires_checkpoint": args.multires_checkpoint
            })
        glue = scglue.models.fit_SCGLUE(
            dataset_dict, prior,
            skip_balance=skip_balance,
            log_wandb=use_wandb,
            init_kws=init_kws,
            compile_kws={"lam_align": lam_align, 
                        "lam_graph": lam_graph,
                        "lam_cycle": lam_cycle,
                        "lam_downsample": args.lam_downsample,
                        "downsample_min": args.downsample_min,
                        "downsample_max": args.downsample_max,
                        "normalize_u": normalize_u,
                        "lr": lr,
                        "modality_weight": modality_weights},
            balance_kws={"resolution": 1.0},
            fit_kws={"directory": "glue" if exp_name is None else exp_name,
                    "neg_samples": neg_samples,
                    "val_split": 0.05,
                    "data_batch_size": batch_size,
                    "max_epochs": max_epochs,
                    "save_interval": save_interval,
                    "wait_n_lrs": wait_n_lrs},
            model=scglue.models.PairedSCGLUEModel if len(coassay) > 0 else scglue.models.SCGLUEModel
        )

        model_name = full_file_suffix if multires else f"{resolution}_{n_genes}_{n_strata}"
        glue.save(f"{out_dir}/glue_hic_{prior_name}_prior_{model_name}.dill")
        if cache_checkpoint:
            os.makedirs(cache_checkpoint, exist_ok=True)
            n_checkpoints = len(os.listdir(cache_checkpoint))
            glue.save(f"{cache_checkpoint}/glue_hic_{prior_name}_prior_{resolution}_{n_checkpoints}.dill")
        # embed and visualize
        rna.obsm["X_glue"] = glue.encode_data("rna", rna)
        hic.obsm["X_glue"] = glue.encode_data("hic", hic)
        if multires:
            # keep the per-resolution parts of the multi-resolution embedding
            for res_name, res_embedding in glue.encode_data_multires("hic", hic).items():
                hic.obsm[f"X_glue_{res_name}"] = res_embedding
                print(f"Per-resolution embedding X_glue_{res_name}: {res_embedding.shape}")
        if atac_file is not None:
            atac.obsm["X_glue"] = glue.encode_data("atac", atac)
        if methyl_file is not None:
            methyl.obsm["X_glue"] = glue.encode_data("methyl", methyl)

        # Metrics and plots (see evals.py). Only embeddings and annotations are
        # used, so this also works for backed data; ground truth labels stay in
        # obs['celltype'] and predictions go to obs['pred_celltype']
        eval_modalities = {"rna": rna, "hic": hic}
        if atac_file is not None:
            eval_modalities["atac"] = atac
        if methyl_file is not None:
            eval_modalities["methyl"] = methyl
        eval_dir = args.eval_dir or os.path.join(exp_name or "glue", "eval")
        metrics, combined = evals.evaluate(
            eval_modalities, eval_dir, label_key="celltype", reference="rna",
            dataset_name=dataset_name, n_neighbors=n_neighbors,
            log_wandb=use_wandb, seed=seed
        )
        print(f"Saved evaluation to {eval_dir}")
        # same modality order as `combined`
        combined_modalities = list(eval_modalities.items())

        if args.gene_list:
            configured_rna_features = pd.Index(rna.uns[scglue.config.ANNDATA_KEY]["features"])
            resolved_genes, missing_genes, ambiguous_genes = _resolve_gene_indices(configured_rna_features, args.gene_list)
            for missing_gene in missing_genes:
                print(f"Skipping gene {missing_gene}: not found in configured RNA decoder features.")
            for ambiguous_gene, matches in ambiguous_genes:
                print(f"Skipping gene {ambiguous_gene}: ambiguous matches {matches}.")

            if resolved_genes:
                gene_plot_dir = f"{out_dir}/predicted_transcription_activity"
                os.makedirs(gene_plot_dir, exist_ok=True)
                gene_indices = [gene_index for _, _, gene_index in resolved_genes]
                decoded_activity = {}
                for modality_name, modality_data in combined_modalities:
                    decoded_activity[modality_name] = _decode_selected_rna_features(
                        glue, modality_name, modality_data, prior, gene_indices, batch_size=batch_size
                    )

                for gene_position, (requested_gene, resolved_gene, _) in enumerate(resolved_genes):
                    combined_gene_activity = np.concatenate([
                        decoded_activity[modality_name][:, gene_position]
                        for modality_name, _ in combined_modalities
                    ])
                    gene_label = resolved_gene if requested_gene == resolved_gene else f"{requested_gene} ({resolved_gene})"
                    output_path = os.path.join(
                        gene_plot_dir,
                        f"{_sanitize_plot_name(requested_gene)}_predicted_transcriptional_activity_umap.png"
                    )
                    if _plot_overlay_umap(
                        combined,
                        combined_gene_activity,
                        f"{gene_label} predicted transcriptional activity",
                        output_path
                    ):
                        print(f"Saved predicted transcriptional activity UMAP for {gene_label} to {output_path}")
                        # if use_wandb:
                        #     wandb.log({
                        #         f"predicted_transcriptional_activity_{_sanitize_plot_name(requested_gene)}": wandb.Image(output_path)
                        #     })

        if cache_checkpoint:
            # save the cell type predictions and cell coordinates
            pred_checkpoint = f"{cache_checkpoint}_predictions"
            os.makedirs(pred_checkpoint, exist_ok=True)
            n_checkpoints = len(os.listdir(pred_checkpoint))
            evals.save_predictions(
                hic, rna, combined, f"{pred_checkpoint}/celltype_prediction_{n_checkpoints}.csv",
                label_map=evals.get_label_map(dataset_name, hic.obs['celltype'], rna.obs['celltype']),
                extra={'resolution': resolution, 'n_genes': n_genes, 'n_strata': n_strata,
                       'filter_strata': filter_strata, 'loop_q': loop_q}
            )

        if args.structure_eval:
            # chromatin structure of pseudobulk pools (structure_eval.py); a failure
            # here should not lose the outputs of a finished training run
            try:
                hic_obs = combined.obs[combined.obs["domain"].astype(str) == "hic"]
                groupings = {"predicted": hic_obs["pred_celltype_harmonized"].astype(str)}
                truth = hic.obs["celltype"].astype(str)
                truth = truth[~truth.isin(evals.UNLABELED)]
                if len(truth) >= args.structure_min_cells:
                    groupings["truth"] = truth
                structure_dir = os.path.join(eval_dir, "structure")
                structure_metrics = structure_eval.run(
                    groupings, args.structure_scool or args.scool, rna, structure_dir,
                    markers=structure_eval.read_markers(args.structure_markers),
                    dataset_name=dataset_name, graph=prior, binsize=args.structure_binsize,
                    min_cells=args.structure_min_cells, max_cells=args.structure_max_cells,
                    top_loci=args.structure_top_loci, seed=seed, log_wandb=use_wandb,
                    # the joint UMAP of the RNA cells, as in the evaluation plots
                    rna_umap=pd.DataFrame(
                        combined.obsm["X_umap"][(combined.obs["domain"].astype(str) == "rna").to_numpy()],
                        index=combined.obs_names[(combined.obs["domain"].astype(str) == "rna").to_numpy()]
                    )
                )
                pd.Series(structure_metrics, name="value").to_csv(
                    os.path.join(structure_dir, "structure_metrics.csv"), header=True)
                print(f"Saved structure analysis to {structure_dir}")
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                print(f"Structure analysis failed: {e}")

        if use_wandb:
            wandb.finish()