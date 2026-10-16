import anndata as ad
import networkx as nx
import scanpy as sc
import scglue
import cooler
from matplotlib import rcParams
from cooler._logging import set_verbosity_level
import os
import sys
import itertools
import argparse
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from multiprocessing import Pool

from tqdm import tqdm
from scipy.linalg import block_diag
from scipy.sparse import coo_matrix, csr_matrix
from networkx.algorithms.bipartite import biadjacency_matrix
from score.sc_args import parse_args
from score.utils.utils import anchor_to_locus, anchor_list_to_dict, sorted_nicely
from score.utils.matrix_ops import VC_SQRT_norm

def OE_norm_old(mat, max_strata=100):
    new_mat = mat.copy() / np.max(mat)  # unchanged values guaranteed to be <=1
    #averages = np.array([np.mean(mat[i:, :len(mat) - i]) for i in range(min(max_strata, len(mat)))])
    averages = np.array([np.mean(np.diagonal(mat, offset=i)) for i in range(min(max_strata, len(mat)))])
    averages = np.where(averages == 0, 1, averages)
    # for i in range(len(mat)):
    #     for j in range(len(mat)):
    #         d = abs(i - j)
    #         if d < max_strata:
    #             new_mat[i, j] = mat[i, j] / averages[d]
    # # vectorized version
    for i in range(min(max_strata, len(mat))):
        new_mat[i:, :len(mat) - i] = mat[i:, :len(mat) - i] / averages[i]
    return new_mat

def OE_norm(mat, max_strata=256, dummy=1e-3, only_original=True):
    """
    Performs distance (strata) based observed/expected (O/E) ratio correction
    on a matrix in a vectorized and efficient manner.

    Args:
        mat (np.ndarray): The input matrix.
        max_strata (int): The maximum diagonal distance to normalize.
        dummy (float): A small value to add to avoid division by zero.
        only_original (bool): If True, only return the original values normalized with no extras

    Returns:
        np.ndarray: The normalized matrix.
    """
    # Ensure input is a NumPy array and handle NaNs
    mat = np.nan_to_num(np.asarray(mat))
    n = mat.shape[0]
    if only_original:
        val_mask = mat > 0

    # Ensure the matrix is square
    if n != mat.shape[1]:
        raise ValueError("Input matrix must be square.")

    # Cap max_strata to the actual number of diagonals in the matrix
    num_diags = min(max_strata, n)

    # Calculate the mean of positive values for each diagonal up to num_diags
    # A list comprehension is a clean way to do this
    averages = np.array([
        np.mean(diag[diag > 0]) if np.any(diag > 0) else 0
        for diag in (np.diagonal(mat, offset=i) for i in range(num_diags))
    ])

    # Replace any calculated zeros with 1 to avoid division by zero
    averages[averages == 0] = 1

    # --- Vectorized creation of the 'expected' matrix ---
    # Create an array of average values indexed by distance from the diagonal.
    # For distances >= num_diags, we will not normalize (i.e., divide by 1).
    avg_by_dist = np.ones(n)
    avg_by_dist[:num_diags] = averages
    # Calculate the distance from the main diagonal for each matrix element
    rows, cols = np.indices(mat.shape)
    dist = np.abs(rows - cols)
    # Build the 'expected' matrix using advanced indexing
    expected_mat = avg_by_dist[dist]
    normalized_mat = (mat + dummy) / (expected_mat + dummy)
    if only_original:
        normalized_mat[~val_mask] = mat[~val_mask]  # keep original values where mask is False
    return normalized_mat

def get_processed_matrix(dataset, cell, cell_i, preprocessing, chr_only=None):
    c = cooler.Cooler(f"{dataset.scool_file}::/cells/{cell}")
    if chr_only is None:
        chr_list = list(pd.unique(dataset.anchor_list['chr'])) if chr_only is None else [chr_only]
        loops = c.pixels()[:]
        chr_mats = []
        for chr_name in chr_list:
            if chr_only is not None and chr_name != chr_only:
                continue
            chr_anchors = dataset.anchor_list.loc[dataset.anchor_list['chr'] == chr_name]
            chr_anchors.reset_index(drop=True, inplace=True)
            chr_anchor_dict = anchor_list_to_dict(chr_anchors['anchor'].values)
            chr_contacts = loops.loc[loops['a1'].isin(chr_anchors['anchor']) & loops['a2'].isin(chr_anchors['anchor'])].copy()
            if len(chr_contacts) > 0:
                chr_contacts['chr1'] = chr_name
                chr_contacts['chr2'] = chr_name

                rows = np.vectorize(anchor_to_locus(chr_anchor_dict))(
                    chr_contacts['a1'].values)  # convert anchor names to row indices
                cols = np.vectorize(anchor_to_locus(chr_anchor_dict))(
                    chr_contacts['a2'].values)  # convert anchor names to column indices
                matrix = coo_matrix((chr_contacts['obs'], (rows, cols)),
                            shape=(len(chr_anchors), len(chr_anchors)))
                mat = matrix.toarray()
            
            else:
                mat = np.zeros((len(chr_anchors), len(chr_anchors)))
            chr_mats.append(mat)
        mat = csr_matrix(block_diag(*chr_mats))
    else:
        mat = c.matrix(sparse=True).fetch(chr_only)
    return cell_i, cell, mat


def get_flattened_matrices(dataset, n_strata, preprocessing=None, agg_fn=None, chr_only=None, offset=0, n_proc=8):
    mats = {}
    results = []
    with Pool(n_proc) as p:
        for cell_i, cell in enumerate(sorted(dataset.cell_list)):
            results.append(p.apply_async(get_processed_matrix, args=(dataset, cell, cell_i, preprocessing, chr_only)))
        for res in tqdm(results):
            cell_i, cell, mat = res.get(timeout=1000)
            if chr_only is not None:
                new_mat = []
                for i in range(n_strata):
                    new_strata = list(mat.diagonal(k=i + offset))
                    if len(new_strata) < mat.shape[0]:
                        new_strata += [0] * (mat.shape[0] - len(new_strata))
                    new_mat.append(new_strata)
                new_mat = np.concatenate(new_mat)
                mats[cell] = new_mat
            else:
                mats[cell] = mat

    full_mats = []
    for cell_i, cell in enumerate(sorted(dataset.cell_list)): 
        full_mats.append(mats[cell])

    if agg_fn is not None:  # aggregating to 1D
        print('Aggregating to 1D')
        strata_mask = np.zeros_like(full_mats[0].toarray())
        for k in range(n_strata):
            strata_mask += np.eye(strata_mask.shape[0], k=k, dtype=full_mats[0].dtype)
        mat = []
        for cell_i, cell in enumerate(sorted(dataset.cell_list)):
            tmp_mat = full_mats[cell_i].toarray()
            tmp_mat[strata_mask == 0] = 0
            mat.append(agg_fn(tmp_mat, axis=0))
    else:  # unraveling 2D strata to 1D vector
        print('Unraveling 2D strata to 1D vector')
        mat = []
        for cell_i, cell in enumerate(sorted(dataset.cell_list)):
            if chr_only is not None:
                counts = full_mats[cell_i]
                mat.append(counts)
            else:
                counts = []
                chr_size = full_mats[cell_i].shape[0]
                for k in range(n_strata):
                    new_strata = list(full_mats[cell_i].diagonal(k=k + offset))
                    print(len(new_strata), chr_size)
                    if len(new_strata) < chr_size:
                        new_strata += [0] * (chr_size - len(new_strata))
                    counts += new_strata
                    
                mat.append(counts)
    x = np.array(mat)
    return x



def load_rna_modality(out_dir, plot_dir, rna_file, gtf_file, n_genes,
                      gene_list=None, load_rna=False, use_xy=False, viz_rna=False):
    """Load, annotate and embed the RNA modality.

    Shared by the single- and multi-resolution preprocessing paths.
    """
    base_rna_filename = 'rna_base_2d.h5ad'
    if  base_rna_filename not in os.listdir(os.path.join(out_dir, 'rna')) or not load_rna:
        rna = ad.read_h5ad(rna_file)
        try:
            rna.X = rna.layers["counts"]
        except Exception as e:
            pass

        if 'batch' not in rna.obs.columns:
            rna.obs['batch'] = 0
        if 'celltype' not in rna.obs.columns:
            if 'cell_type' in rna.obs.columns:
                rna.obs['celltype'] = rna.obs['cell_type']
            else:
                rna.obs['celltype'] = 'Unknown'
        rna.layers["counts"] = rna.X.copy()
        keep_columns = scglue.genomics.Bed.COLUMNS
        for col in keep_columns:
            try:
                rna.var.drop(columns=[col], inplace=True)
            except Exception as e:
                print(e)
                pass
        try:
            scglue.data.get_gene_annotation(
                rna, gtf=gtf_file,
                gtf_by="gene_name"
            )
        except KeyError:
            scglue.data.get_gene_annotation(
                rna, gtf=gtf_file,
                gtf_by="gene_symbol"
            )


        rna.var['chromStart'] = rna.var['chromStart'].fillna(0).astype(int)
        rna.var['chromEnd'] = rna.var['chromEnd'].fillna(0).astype(int)
        rna.var['strand'] = rna.var['strand'].fillna('+')
        rna = rna[:, rna.var['chrom'].notna()].copy()
        try:
            if not rna.var['chrom'].iloc[0].startswith('chr'):
                rna.var['chrom'] = 'chr' + rna.var['chrom']
        except AttributeError:
            print(rna.var['chrom'].iloc[0])
            print(rna.var['chrom'])

        drop_cols = []
        for col in rna.var.columns:
            if col not in keep_columns:
                drop_cols.append(col)
        rna.var.drop(columns=drop_cols, inplace=True)
        rna = rna[:, rna.var['chrom'].notna()].copy()
        if not use_xy:
            rna = rna[:, ~rna.var['chrom'].str.lower().str.contains('x|y')].copy()
        genes = scglue.genomics.Bed(rna.var.assign(name=rna.var_names))
        rna.write(f"{out_dir}/rna/{base_rna_filename}", compression="gzip")
    else:
        rna = ad.read_h5ad(f"{out_dir}/rna/{base_rna_filename}")

    sc.pp.filter_genes(rna, min_counts=1)
    print('Embedding RNA...')
    print('Highly variable genes...')

    sc.pp.highly_variable_genes(rna, n_top_genes=n_genes, flavor="seurat_v3", span=1)
    if gene_list is not None:
        print('Adding user provided gene list to highly variable genes...')
        for gene in gene_list:
            if gene in rna.var_names:
                rna.var.loc[gene, 'highly_variable'] = True

    # # #sc.pp.highly_variable_genes(rna, min_mean=0.0125, max_mean=3, min_disp=0.5)
    print("Normalize")
    sc.pp.normalize_total(rna)
    sc.pp.log1p(rna)
    sc.pp.scale(rna)
    print("PCA")
    sc.tl.pca(rna, n_comps=200, svd_solver="auto")
    if viz_rna:
        sc.pp.neighbors(rna, n_pcs=200, metric="cosine")
        sc.tl.umap(rna)

        fig = sc.pl.umap(rna, color=["celltype", "batch"], return_fig=True, wspace=0.6)
        fig.tight_layout()
        fig.savefig(f"{plot_dir}/rna_umap.png")
        plt.close()
    return rna


def load_atac_modality(atac_file, plot_dir, n_atac_peaks, n_hic_features):
    """Load, annotate and embed the ATAC modality."""
    atac = ad.read_h5ad(atac_file)
    if 'batch' not in atac.obs.columns:
        atac.obs['batch'] = 0
    if 'celltype' not in atac.obs.columns:
        if 'cell_type' in atac.obs.columns:
            atac.obs['celltype'] = atac.obs['cell_type']
        else:
            atac.obs['celltype'] = 'Unknown'
    atac.layers["counts"] = atac.X.copy()
    atac.var['chrom'] = atac.var_names.map(lambda s: s.split(':')[0])
    atac.var['chromStart'] = atac.var_names.map(lambda s: s.split(':')[1].split('-')[0]).astype(int)
    atac.var['chromEnd'] = atac.var_names.map(lambda s: s.split(':')[1].split('-')[1]).astype(int)
    atac.var['name'] = atac.var_names
    atac = atac[:, atac.var['chrom'].notna()].copy()
    # only accept chroms 1-22, X and Y
    atac = atac[:, atac.var['chrom'].isin([f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY'])].copy()
    atac_peaks = scglue.genomics.Bed(atac.var.assign(name=atac.var_names))
    sc.pp.filter_genes(atac, min_counts=2)
    # use same number of features as in Hi-C data
    sc.pp.highly_variable_genes(atac, n_top_genes=n_atac_peaks if n_atac_peaks is not None else n_hic_features, flavor="seurat_v3", span=1)
    sc.pp.normalize_total(atac)
    sc.pp.log1p(atac)
    sc.pp.scale(atac)
    sc.tl.pca(atac, n_comps=200, svd_solver="auto")
    sc.pp.neighbors(atac, n_pcs=200, metric="cosine")
    sc.tl.umap(atac)
    fig = sc.pl.umap(atac, color=["celltype", "batch"], return_fig=True, wspace=0.6)
    fig.tight_layout()
    fig.savefig(f"{plot_dir}/atac_umap.png")
    plt.close()
    return atac, atac_peaks


def load_methyl_modality(methyl_file, rna, gtf_file, n_genes, plot_dir):
    """Load, annotate and embed the methylation modality."""
    methyl = ad.read_h5ad(methyl_file)
    if 'batch' not in methyl.obs.columns:
        methyl.obs['batch'] = 0
    methyl.layers["counts"] = methyl.X.copy()
    methyl.var['name'] = methyl.var_names
    methyl.var['name'] = methyl.var['name'].apply(lambda s: s.replace('_mCH', '').replace('_mCG', ''))
    keep_columns = scglue.genomics.Bed.COLUMNS
    for col in keep_columns:
        try:
            rna.var.drop(columns=[col], inplace=True)
        except Exception as e:
            print(e)
            pass
    try:
        scglue.data.get_gene_annotation(
            rna, gtf=gtf_file,
            gtf_by="gene_name"
        )
    except KeyError:
        scglue.data.get_gene_annotation(
            rna, gtf=gtf_file,
            gtf_by="gene_symbol"
        )
    methyl.var['chromStart'] = methyl.var['chromStart'].fillna(0).astype(int)
    methyl.var['chromEnd'] = methyl.var['chromEnd'].fillna(0).astype(int)
    methyl.var['strand'] = methyl.var['strand'].fillna('+')
    methyl = methyl[:, methyl.var['chrom'].notna()].copy()
    # only accept chroms 1-22, X and Y
    methyl = methyl[:, methyl.var['chrom'].isin([f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY'])].copy()
    # add _mCH suffix to var names to distinguish from RNA genes
    #methyl.var_names = methyl.var_names.map(lambda s: s + '_mCH')
    # only keep methylation for genes in RNA data
    methyl = methyl[:, methyl.var['name'].isin(rna.var_names)].copy()
    methyl_genes = scglue.genomics.Bed(methyl.var.assign(name=methyl.var_names))
    sc.pp.filter_genes(methyl, min_counts=2)
    # use same number of genes as in RNA data
    sc.pp.highly_variable_genes(methyl, n_top_genes=n_genes, flavor="seurat_v3", span=1)
    sc.pp.normalize_total(methyl)
    sc.pp.log1p(methyl)
    sc.pp.scale(methyl)
    sc.tl.pca(methyl, n_comps=200, svd_solver="auto")
    sc.pp.neighbors(methyl, n_pcs=200, metric="cosine")
    sc.tl.umap(methyl)
    fig = sc.pl.umap(methyl, color=["celltype", "batch"], return_fig=True, wspace=0.6)
    fig.tight_layout()
    fig.savefig(f"{plot_dir}/methyl_umap.png")
    plt.close()
    return methyl, methyl_genes


def multires_extra_suffix(use_dist_norm, use_trans):
    """File name marker for pseudobulk edge options in multi-resolution mode."""
    parts = []
    if use_dist_norm:
        parts.append('oe')
    if use_trans:
        parts.append('trans')
    return '-'.join(parts)


def _add_edges(graph, edges, edge_type, sign=1, symmetric=True):
    """Add an edge table (``source``/``target``/``weight``/``dist``) to a graph."""
    for row in edges.itertuples(index=False):
        graph.add_edge(
            row.source, row.target, weight=float(row.weight), sign=sign,
            type=edge_type, dist=float(getattr(row, "dist", 0.0))
        )
        if symmetric:
            graph.add_edge(
                row.target, row.source, weight=float(row.weight), sign=sign,
                type=edge_type, dist=float(getattr(row, "dist", 0.0))
            )
    return graph


def preprocess_higlue_multires(args, glue_args):
    """Build multi-resolution Hi-C features and a multi-scale guidance graph.

    Unlike :func:`preprocess_higlue`, cells are streamed from the ``.scool``
    file one at a time and the feature space of every resolution is filtered
    down to a budget of informative anchors *before* any per-cell data is
    written, so neither preprocessing nor training needs the full dense matrix
    in memory.
    """
    import multires_hic as mh

    resolutions = list(args.resolutions)
    binsizes = [mh.parse_resolution(res) for res in resolutions]
    order = np.argsort(binsizes)[::-1]  # coarse to fine
    resolutions = [resolutions[i] for i in order]
    n_res = len(resolutions)
    strata_list = mh.broadcast_option(
        args.multires_strata, n_res, "multires_strata", default=args.n_strata
    )
    anchors_list = mh.broadcast_option(
        args.multires_max_anchors, n_res, "multires_max_anchors", default=50000
    )
    n_strata_map = {res: int(s) for res, s in zip(resolutions, strata_list)}
    max_anchors_map = {res: int(a or 0) for res, a in zip(resolutions, anchors_list)}
    loop_q = float(args.loop_q)
    use_trans = args.use_trans
    use_dist_norm = args.use_dist_norm
    use_xy = args.use_xy
    n_genes = args.n_genes
    gtf_file = args.gtf
    rna_file = args.rna_file
    atac_file = args.atac_file
    methyl_file = args.methyl_file
    dataset_name = args.dset
    out_dir = f'data/{dataset_name}_data'
    if args.data_dir != '':
        out_dir = f'{args.data_dir}'
    plot_dir = f'plots/{dataset_name}_plots'
    for sub in ('rna', 'hic', 'atac', 'methyl', 'graphs'):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    print('Parsing SCORE args...')
    parser = argparse.ArgumentParser()
    sys.argv = sys.argv[glue_args - 1:]
    score_args, _, _, _, _, dataset, _ = parse_args(parser)

    print('Loading RNA...')
    rna = load_rna_modality(
        out_dir, plot_dir, rna_file, gtf_file, n_genes,
        gene_list=args.gene_list, load_rna=args.load_rna,
        use_xy=use_xy, viz_rna=args.viz_rna
    )
    genes = scglue.genomics.Bed(rna.var.assign(name=rna.var_names))
    promoters = genes.strand_specific_start_site().expand(2000, 0)
    hv_promoters = promoters.loc[promoters.index.isin(
        rna.var.query("highly_variable").index
    )]
    print(f'{rna.shape[1]} genes ({rna.var["highly_variable"].sum()} highly variable)')

    reader = mh.ScoolReader(dataset.scool_file, res_name=dataset.res_name)
    try:
        base_bins = reader.bins()
        chromsizes = reader.chromsizes()
        cells = sorted(dataset.cell_list)
        print(f'{len(cells)} cells, {base_bins.shape[0]} source bins '
              f'at {dataset.res_name}')

        grids = mh.build_grids(
            base_bins, chromsizes, resolutions,
            base_resolution=args.resolution, use_xy=use_xy
        )
        for res in resolutions:
            print(f'  {res}: {grids[res].n_bins:,} bins, '
                  f'{n_strata_map[res]} strata '
                  f'(up to {n_strata_map[res] * mh.parse_resolution(res) / 1e6:.2f} Mb)')

        print('Collecting band statistics...')
        # trans contacts are only summarized at the coarsest resolution: a
        # dense trans matrix is quadratic in the number of bins
        trans_res = resolutions[0] if use_trans else None
        if trans_res is not None:
            print(f'  also collecting trans contacts at {trans_res}')
        stats, trans_stats = mh.collect_band_stats(
            reader, cells, grids, n_strata_map,
            max_cells=args.multires_stat_cells, random_state=args.seed,
            trans_res=trans_res
        )
        min_cells = max(3, int(args.multires_min_frac * stats[resolutions[0]].n_cells))

        anchors = {}
        for res in resolutions:
            forced = mh.mark_overlapping_bins(grids[res], hv_promoters)
            anchors[res] = mh.select_anchors(
                stats[res], max_anchors=max_anchors_map[res] or None,
                min_cells=min_cells, tile_size=args.multires_tile_size,
                forced=forced
            )
            print(f'  {res}: {anchors[res].size:,} of {grids[res].n_bins:,} anchors '
                  f'({int(forced.sum()):,} overlap a variable gene promoter) '
                  f'-> {anchors[res].size * n_strata_map[res]:,} features')

        print('Building multi-scale guidance graph...')
        prior = nx.MultiDiGraph()
        peaks = {}
        overlap_graphs = {}
        for res in resolutions:
            peaks[res] = scglue.genomics.Bed(mh.anchor_bed(grids[res], anchors[res]))
            overlap_graphs[res] = scglue.genomics.window_graph(
                promoters, peaks[res], 0,
                attr_fn=lambda l, r, d: {"weight": 1.0, "type": "overlap", "sign": 1, "dist": 0.0}
            )
            prior = scglue.graph.compose_multigraph(
                prior,
                scglue.graph.compose_multigraph(
                    overlap_graphs[res], overlap_graphs[res].reverse()
                )
            )
            print(f'  {res}: {overlap_graphs[res].number_of_edges():,} promoter overlap edges')
            if args.multires_use_dist:
                dist_graph = scglue.genomics.window_graph(
                    promoters, peaks[res], args.multires_dist_window,
                    attr_fn=lambda l, r, d: {
                        "weight": scglue.genomics.dist_power_decay(abs(d)),
                        "type": "dist", "sign": 1, "dist": abs(d) / 2e6
                    }
                )
                prior = scglue.graph.compose_multigraph(
                    prior,
                    scglue.graph.compose_multigraph(dist_graph, dist_graph.reverse())
                )
                print(f'  {res}: {dist_graph.number_of_edges():,} promoter distance edges')

            loops = mh.loop_edges(
                stats[res], anchors[res], loop_q=loop_q,
                coverage_norm=use_dist_norm
            )
            _add_edges(prior, loops, "hic")
            adjacent = mh.adjacency_edges(grids[res], anchors[res])
            _add_edges(prior, adjacent, "hic")
            print(f'  {res}: {loops.shape[0]:,} contact edges, '
                  f'{adjacent.shape[0]:,} adjacency edges')
            if trans_stats is not None and res == trans_res:
                trans = mh.trans_edges(
                    trans_stats, anchors[res], loop_q=loop_q,
                    coverage_norm=use_dist_norm
                )
                _add_edges(prior, trans, "trans")
                print(f'  {res}: {trans.shape[0]:,} trans contact edges')

        for coarse, fine in zip(resolutions[:-1], resolutions[1:]):
            links = mh.hierarchy_edges(
                grids[fine], anchors[fine], grids[coarse], anchors[coarse]
            )
            _add_edges(prior, links, "hierarchy")
            print(f'  {fine} -> {coarse}: {links.shape[0]:,} hierarchy edges')

        atac = atac_peaks = methyl = methyl_genes = None
        if atac_file is not None:
            n_hic_features = sum(
                anchors[res].size * n_strata_map[res] for res in resolutions
            )
            atac, atac_peaks = load_atac_modality(
                atac_file, plot_dir, args.n_atac_peaks, n_hic_features
            )
            atac_peaks = scglue.genomics.Bed(atac_peaks.assign(name=atac_peaks['name']))
            atac_graph = scglue.genomics.window_graph(
                promoters, atac_peaks, 0,
                attr_fn=lambda l, r, d: {"weight": 1.0, "type": "overlap", "sign": 1, "dist": 0.0}
            )
            prior = scglue.graph.compose_multigraph(
                prior, scglue.graph.compose_multigraph(atac_graph, atac_graph.reverse())
            )
            for res in resolutions:
                atac_hic_graph = scglue.genomics.window_graph(
                    atac_peaks, peaks[res], 0,
                    attr_fn=lambda l, r, d: {"weight": 1.0, "type": "overlap", "sign": 1, "dist": 0.0}
                )
                prior = scglue.graph.compose_multigraph(
                    prior,
                    scglue.graph.compose_multigraph(atac_hic_graph, atac_hic_graph.reverse())
                )
            print(f'  ATAC: {atac_graph.number_of_edges():,} promoter overlap edges')
        if methyl_file is not None:
            methyl, methyl_genes = load_methyl_modality(
                methyl_file, rna, gtf_file, n_genes, plot_dir
            )
            methyl_genes = scglue.genomics.Bed(methyl_genes.assign(name=methyl_genes['name']))
            methyl_promoters = methyl_genes.strand_specific_start_site().expand(2000, 0)
            methyl_graph = scglue.genomics.window_graph(
                promoters, methyl_promoters, 0,
                attr_fn=lambda l, r, d: {"weight": 1.0, "type": "overlap", "sign": -1, "dist": 0.0}
            )
            prior = scglue.graph.compose_multigraph(
                prior, scglue.graph.compose_multigraph(methyl_graph, methyl_graph.reverse())
            )
            for res in resolutions:
                methyl_hic_graph = scglue.genomics.window_graph(
                    methyl_promoters, peaks[res], 0,
                    attr_fn=lambda l, r, d: {"weight": 1.0, "type": "overlap", "sign": -1, "dist": 0.0}
                )
                prior = scglue.graph.compose_multigraph(
                    prior,
                    scglue.graph.compose_multigraph(methyl_hic_graph, methyl_hic_graph.reverse())
                )
            print(f'  Methyl: {methyl_graph.number_of_edges():,} promoter overlap edges')

        anchor_names = set()
        for res in resolutions:
            anchor_names.update(grids[res].names[anchors[res]])
        rna.var["in_hic"] = [
            node in prior and any(
                neighbor in anchor_names for neighbor in prior.neighbors(node)
            ) for node in rna.var_names
        ]
        print('Genes linked to Hi-C anchors:', int(np.sum(rna.var["in_hic"])))

        for node in list(prior.nodes):
            if not prior.has_edge(node, node):
                prior.add_edge(node, node, weight=1.0, type="self-loop", sign=1, dist=0.0)

        reachable = scglue.graph.reachable_vertices(
            prior, rna.var.query("highly_variable").index
        )
        prior = nx.MultiDiGraph(prior.subgraph(reachable))
        print(f'Guidance graph: {prior.number_of_nodes():,} nodes, '
              f'{prior.number_of_edges():,} edges')

        # Anchors that the graph cannot reach from a variable gene are dropped
        # here, before any per-cell data is written.
        for res in resolutions:
            names = grids[res].names[anchors[res]]
            keep = np.array([name in reachable for name in names])
            if not keep.all():
                print(f'  {res}: dropping {int((~keep).sum()):,} unreachable anchors')
                anchors[res] = anchors[res][keep]
            if not anchors[res].size:
                raise ValueError(
                    f"No anchor of resolution '{res}' is reachable from the "
                    f"highly variable genes!"
                )
        if atac is not None:
            atac.var["highly_variable"] = atac.var_names.isin(reachable) \
                & atac.var["highly_variable"]
            print('ATAC features in graph:', int(atac.var["highly_variable"].sum()))
        if methyl is not None:
            methyl.var["highly_variable"] = methyl.var_names.isin(reachable) \
                & methyl.var["highly_variable"]
            print('Methyl features in graph:', int(methyl.var["highly_variable"].sum()))

        layout = mh.MultiResLayout(grids, anchors, n_strata_map, resolutions)
        print(f'Total Hi-C features: {layout.n_features:,}')

        reference = dataset.reference
        obs = pd.DataFrame({
            "celltype": np.array([str(reference.loc[cell, 'cluster']) for cell in cells]),
            "depth": np.array([float(reference.loc[cell, 'depth']) for cell in cells]),
            # the batch dtype is kept as-is (numeric batches stay numeric, as in
            # the single-resolution pipeline)
            "batch": np.array([
                reference.loc[cell, 'batch'] for cell in cells
            ]) if 'batch' in reference.columns else np.zeros(len(cells), dtype=int),
            "dataset": "train"
        }, index=pd.Index([
            cell.replace(f'.{dataset.res_name}', '') for cell in cells
        ]))

        suffix = mh.multires_suffix(
            resolutions, [n_strata_map[r] for r in resolutions], loop_q,
            extra=multires_extra_suffix(use_dist_norm, use_trans)
        )
        hic_path = f"{out_dir}/hic/hic_{suffix}.h5ad"
        print(f'Streaming cells into {hic_path} ...')
        depths = mh.write_multires_hic(
            hic_path, reader, cells, layout, obs,
            chunk_size=args.multires_chunk_size,
            uns={"multires_res_order": resolutions}
        )
        for res in resolutions:
            print(f'  {res}: mean in-band contacts per cell {depths[res].mean():,.1f}')
    finally:
        reader.close()

    chrom_attr, pos_attr, type_attr = {}, {}, {}
    for node in prior.nodes:
        if node in genes.index:
            type_attr[node] = 'RNA'
            chrom_attr[node] = str(genes.loc[node, 'chrom'])
            pos_attr[node] = int(genes.loc[node, 'chromStart'])
        elif atac is not None and node in atac.var_names:
            type_attr[node] = 'ATAC'
            chrom_attr[node] = str(atac_peaks.loc[node, 'chrom'])
            pos_attr[node] = int(atac_peaks.loc[node, 'chromStart'])
        elif methyl is not None and node in methyl.var_names:
            type_attr[node] = 'Methyl'
            chrom_attr[node] = str(methyl_genes.loc[node, 'chrom'])
            pos_attr[node] = int(methyl_genes.loc[node, 'chromStart'])
        else:
            type_attr[node] = 'Hi-C'
            chrom, _, pos = str(node).partition(':')
            chrom_attr[node] = chrom
            try:
                pos_attr[node] = int(pos.split('-')[0])
            except ValueError:
                pos_attr[node] = 0
    nx.set_node_attributes(prior, chrom_attr, "chrom")
    nx.set_node_attributes(prior, pos_attr, "chrom_pos")
    nx.set_node_attributes(prior, type_attr, "feature_type")
    edge_ids = {edge: i for i, edge in enumerate(prior.edges)}
    nx.set_edge_attributes(prior, edge_ids, "edge_id")

    rna.write(f"{out_dir}/rna/rna_{suffix}.h5ad", compression="gzip")
    if atac is not None:
        atac.write(f"{out_dir}/atac/atac_{suffix}.h5ad", compression="gzip")
    if methyl is not None:
        methyl.write(f"{out_dir}/methyl/methyl_{suffix}.h5ad", compression="gzip")
    nx.write_graphml(
        prior, f"{out_dir}/graphs/{args.prior}_prior_{suffix}.graphml.gz",
        edge_id_from_attribute='edge_id', named_key_ids=True
    )
    print(f'Wrote {out_dir}/graphs/{args.prior}_prior_{suffix}.graphml.gz')
    return suffix


def preprocess_higlue(args, glue_args):
    n_strata = args.n_strata
    resolution = args.resolution
    gtf_file = args.gtf
    rna_file = args.rna_file
    atac_file = args.atac_file
    methyl_file = args.methyl_file
    loop_q = args.loop_q
    load_rna = args.load_rna
    load_hic = False
    use_toploops = False 
    use_ice = args.use_ice
    use_dist_norm = args.use_dist_norm
    use_trans = args.use_trans
    viz_rna = args.viz_rna
    use_raw_pseudobulk = True
    loops_offset = args.offset
    min_count = args.min_count
    n_distal_interactions = args.distal_interactions
    filter_strata = args.filter_strata
    exclusive_strata = args.exclusive_strata
    use_xy = args.use_xy
    n_genes = args.n_genes
    n_atac_peaks = args.n_atac_peaks
    gene_list = args.gene_list
    bulk_hic = args.bulk_hic
    if not use_toploops:
        loop_q = float(loop_q)
    else:
        if loop_q.endswith('k'):
            n_loops = int(loop_q[:-1]) * 1000
    dataset_name = args.dset
    out_dir = f'data/{dataset_name}_data'
    plot_dir = f'plots/{dataset_name}_plots'
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'rna'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'graphs'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'hic'), exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    print('Parsing SCORE args...')
    parser = argparse.ArgumentParser()
    sys.argv = sys.argv[glue_args - 1:]
    args, x, y, depths, batches, dataset, valid_dataset = parse_args(parser)

    if bulk_hic is not None:
        bulk = cooler.Cooler(bulk_hic)
    else:
        cool_files = cooler.fileops.list_scool_cells(dataset.scool_file)
        cool_files = [f"{dataset.scool_file}::{cell}" for cell in cool_files]
        os.makedirs('data/scools', exist_ok=True)
        out_cool_file = f"data/scools/{dataset_name}_{resolution}_bulk.cool"

        try:
            bulk = cooler.Cooler(out_cool_file)
            if use_ice:
                if 'weight' not in bulk.bins().columns:
                    cooler.balance_cooler(bulk, cis_only=True, store=True)
        except Exception as e:
            print('Could not find bulk cooler file, generating from psuedobulk...')
            set_verbosity_level(1)
            cooler.merge_coolers(out_cool_file, cool_files, mergebuf=40000000)
            bulk = cooler.Cooler(out_cool_file)
            if use_ice:
                cooler.balance_cooler(bulk, cis_only=False, store=True)
                bulk = cooler.Cooler(out_cool_file)
    # plot example heatmap
    mat = bulk.matrix(balance=use_ice).fetch('chr10')
    midpoint = mat.shape[0] // 2
    mat = mat[midpoint - n_strata:midpoint + n_strata, midpoint - n_strata:midpoint + n_strata]
    plt.imshow(mat, cmap='Reds', norm=LogNorm())
    plt.colorbar()
    plt.savefig(f'{plot_dir}/hic_example.png')
    plt.close()

    frags = bulk.bins()[:]
    frags.rename(columns={'start': 'chromStart', 'end': 'chromEnd'}, inplace=True)
    frags['name'] = frags.index.astype(str)
    loops = bulk.pixels(join=False)[:]
    chr_map = frags['chrom'].to_dict()
    start_map = frags['chromStart'].to_dict()
    end_map = frags['chromEnd'].to_dict()
    loops['chr1'] = loops['bin1_id'].map(chr_map)   
    loops['chr2'] = loops['bin2_id'].map(chr_map)

    if use_trans:
        # use total of same loop_q trans loops by transforming based on number of chroms
        trans_loop_q = 1 - (1 - loop_q) / (len(bulk.chromnames) * (len(bulk.chromnames) - 1) / 2)
        print(f"Trans loop quantile:{trans_loop_q:.4f}")
        trans = []
        used_chrs = []
        print('Loading trans chromosomal interactions...')
        for chrom1 in tqdm(bulk.chromnames):
            chr_trans_pixels = []
            for chrom2 in bulk.chromnames:
                if chrom1 == chrom2:
                    continue
                if (chrom1, chrom2) in used_chrs or (chrom2, chrom1) in used_chrs:
                    continue
                used_chrs.append((chrom1, chrom2))
                #trans_pixels = bulk.pixels(join=False).fetch(chrom1, chrom2)
                trans_pixels = loops[(loops['chr1'] == chrom1) & (loops['chr2'] == chrom2)].copy()
                trans_pixels.drop(columns=['chr1', 'chr2'], inplace=True)
                chr_trans_pixels.append(trans_pixels)
            if len(chr_trans_pixels) > 0:
                chr_trans_pixels = pd.concat(chr_trans_pixels).reset_index(drop=True)
                chr_trans_pixels['rank'] = chr_trans_pixels['count'].rank(pct=True)
                try:
                    loop_cutoff = np.quantile(chr_trans_pixels['rank'].values, q=trans_loop_q)
                    chr_trans_pixels = chr_trans_pixels.loc[chr_trans_pixels['rank'] >= loop_cutoff].copy()
                    chr_trans_pixels['rank'] = chr_trans_pixels['rank'].rank(pct=True)
                    #chr_trans_pixels['rank'] = chr_trans_pixels['rank'] * 0.5 + 0.5
                    trans.append(chr_trans_pixels)
                except IndexError:  # no reads in chrom (e.g no chrY)
                    pass  
        trans = pd.concat(trans).reset_index(drop=True)
        print('Trans interactions:')
        print(trans)
    if exclusive_strata:
        # remove interactions beyond n_strata
        loops['strata'] = abs(loops['bin1_id'] - loops['bin2_id'])
        loops = loops[loops['strata'] <= n_strata].copy()
        loops.drop(columns=['strata'], inplace=True)

    if use_ice:
        weight_map = frags['weight'].to_dict()
        loops['weight1'] = loops['bin1_id'].map(weight_map)
        loops['weight2'] = loops['bin2_id'].map(weight_map)
        loops['oe'] = loops['count'] * loops['weight1'] * loops['weight2']
        loops['rank'] = loops['oe'].rank(pct=True)
    else:
        loops['rank'] = loops['count'].rank(pct=True)
    loops.dropna(inplace=True)

    if use_dist_norm:
        # extract each dense matrix and compute OE normalization
        print('OE normalizing Hi-C data...')
        oe_loops = []
        for chr_name in tqdm(sorted_nicely(bulk.chromnames)):
            if not use_xy and ('x' in chr_name.lower() or 'y' in chr_name.lower()):
                continue
            chr_loops = loops[(loops['chr1'] == chr_name) & (loops['chr2'] == chr_name)].copy()
            chr_anchors = frags[frags['chrom'] == chr_name].copy()
            chr_offset = chr_anchors.index[0]
            chr_anchors.reset_index(drop=True, inplace=True)
            
            # convert to dense matrix
            if use_ice:
                val_col = 'oe'
            else:
                val_col = 'count'
            chr_mat = csr_matrix((chr_loops[val_col].values, 
                                  (chr_loops['bin1_id'].values - chr_offset, chr_loops['bin2_id'].values - chr_offset)), 
                                  shape=(len(chr_anchors), len(chr_anchors)))
            # chr_mat = bulk.matrix(balance=use_ice).fetch(chr_name)
            chr_mat = OE_norm(VC_SQRT_norm(chr_mat.toarray()))
            chr_mat = csr_matrix(chr_mat)
            oe_values = chr_mat.data
            chr_loops['oe'] = oe_values
            oe_loops.append(chr_loops)
            if chr_name == 'chr10':  # visualize first matrix
                mat = chr_mat.toarray()
                mat = mat + mat.T - np.diag(np.diag(mat))
                midpoint = mat.shape[0] // 2
                mat = mat[midpoint - n_strata:midpoint + n_strata, midpoint - n_strata:midpoint + n_strata]
                plt.imshow(mat, cmap='Reds', norm=LogNorm())
                plt.colorbar()
                plt.savefig(f'{plot_dir}/hic_example_oe.png')
                plt.close()
        loops = pd.concat(oe_loops).reset_index(drop=True)
        loops['rank'] = loops['oe'].rank(pct=True)
        print(loops)
    
    loop_dfs = []
    print('Filtering top loops in each chromosome...')
    for chr_name in tqdm(sorted_nicely(bulk.chromnames)):
        if not use_xy and ('x' in chr_name.lower() or 'y' in chr_name.lower()):
            continue
        chr_loops = loops[(loops['chr1'] == chr_name) & (loops['chr2'] == chr_name)].copy()
        chr_loops['rank'] = chr_loops['oe' if use_ice or use_dist_norm else 'count'].rank(pct=True)
        loop_cutoff = np.quantile(chr_loops['rank'].values, q=loop_q)
        chr_loops = chr_loops.loc[chr_loops['rank'] >= loop_cutoff].copy()
        chr_loops['rank'] = chr_loops['rank'].rank(pct=True)
        chr_loops.reset_index(drop=True, inplace=True)
        chr_loops.drop(columns=['chr1', 'chr2'], inplace=True)
        if use_ice:
            chr_loops.drop(columns=['weight1', 'weight2', 'oe'], inplace=True)
        loop_dfs.append(chr_loops)
    loops = pd.concat(loop_dfs).reset_index(drop=True)
    
    if use_trans:
        loops = pd.concat([loops, trans]).reset_index(drop=True)

    print('Final loops:')
    print(loops)
    if not use_xy:
        frags = frags[~frags['chrom'].str.lower().str.contains('x|y')].reset_index(drop=True)
    frags.rename(columns={'start': 'chromStart', 'end': 'chromEnd'}, inplace=True)

    rna = load_rna_modality(
        out_dir, plot_dir, rna_file, gtf_file, n_genes,
        gene_list=gene_list, load_rna=load_rna, use_xy=use_xy, viz_rna=viz_rna
    )

    base_hic_filename = f"hic_base_{resolution}_2d.h5ad"
    base_hic_path = f"{out_dir}/hic/{base_hic_filename}"
    if base_hic_filename not in os.listdir(os.path.join(out_dir, 'hic')) or not load_hic:
        print('Loading scHi-C sparse matrices...')
        mats = dataset.get_sparse_matrices()
        mat = []
        strata_adatas = []
        total_interactions = 0
        print(f'Processing strata...')
        for k in range(abs(loops_offset), n_strata + abs(loops_offset)):
            strata_mat = []
            for cell_i, cell in enumerate(sorted(dataset.cell_list)):
                new_strata = list(mats[cell_i].diagonal(k=k))
                if len(new_strata) < len(dataset.anchor_list):
                    new_strata += [0] * (len(dataset.anchor_list) - len(new_strata))
                if resolution in ['10kb', '20kb', '50kb']:
                    strata_mat.append(np.uint8(new_strata))
                else:
                    strata_mat.append(new_strata)
            # create per-strata anndata
            strata_hic = ad.AnnData(np.array(strata_mat), dtype=np.uint8 if resolution in ['10kb', '20kb', '50kb'] else np.int32)
            strata_hic.obs_names = sorted(dataset.cell_list)
            strata_hic.obs_names = strata_hic.obs_names.map(lambda s: s.replace(f'.{dataset.res_name}', ''))
            genomic_pos = dataset.anchor_list.apply(lambda row: f"{row['chr']}:{row['start']}-{row['end']}", axis=1)
            # if not use_xy:
            #     genomic_pos = genomic_pos[~genomic_pos.str.lower().str.contains('x|y')].reset_index(drop=True)
            if k - loops_offset == 0:
                strata_hic.var_names = genomic_pos
            else:
                if strata_hic.var_names.shape != genomic_pos.shape:
                    strata_hic.var_names = genomic_pos.iloc[:-(k - loops_offset)] + f'-{k - loops_offset}'
                else:
                    strata_hic.var_names = genomic_pos + f'-{k - loops_offset}'
            strata_hic.var['root'] = strata_hic.var_names.map(lambda s: s.rsplit('-', 1)[0] if (s[-2] == '-' or s[-3] == '-') else s)
            split = strata_hic.var['root'].str.split(r"[:-]")
            strata_hic.var["chrom"] = split.map(lambda x: x[0])
            strata_hic.var["chromStart"] = split.map(lambda x: x[1]).astype(int)
            strata_hic.var["chromEnd"] = split.map(lambda x: x[2]).astype(int)
            if not use_xy:
                strata_hic = strata_hic[:, ~strata_hic.var['chrom'].str.lower().str.contains('x|y')].copy()
            next_strata_map = frags.copy()
            next_strata_map['name'] = next_strata_map.apply(lambda row: f"{row['chrom']}:{row['chromStart']}-{row['chromEnd']}", axis=1)
            roots = next_strata_map['name'].values
            next_vars = np.roll(roots, shift=-k)
            next_vars[-k:] = roots[-k:]
            next_strata_map['target'] = next_vars
            next_strata_map = next_strata_map.set_index('name').to_dict()['target']

            strata_hic.var['target'] = strata_hic.var['root'].map(next_strata_map)
            split = strata_hic.var['target'].str.split(r"[:-]")
            strata_hic.var["target_chrom"] = split.map(lambda x: x[0])
            strata_hic.var["target_chromStart"] = split.map(lambda x: x[1]).astype(int)
            strata_hic.var["target_chromEnd"] = split.map(lambda x: x[2]).astype(int)

            if k - loops_offset > 0:  # keep all features for the first strata as roots for later, even if they have no interactions
                sc.pp.filter_genes(strata_hic, min_counts=min_count)
                sc.pp.filter_genes(strata_hic, min_cells=min_count)
                if filter_strata:
                    # find highly variable features
                    sc.pp.highly_variable_genes(strata_hic, n_top_genes=int(strata_hic.shape[1] * filter_strata), flavor="seurat_v3", span=1)
                    top_loop_mask = strata_hic.var['highly_variable'].values
                    strata_hic = strata_hic[:, top_loop_mask].copy()
            strata_hic.var.drop(columns=['root'], inplace=True)
            strata_adatas.append(strata_hic)
            total_interactions += strata_hic.shape[1]
            print(f'{k} - Total interactions: {total_interactions:,} ({strata_hic.shape[1]:,} from strata {k})')
        
        hic = ad.concat(strata_adatas, axis=1, join='inner')
        hic.layers["counts"] = hic.X.copy()
        hic.obs['celltype'] = np.array([dataset.reference.loc[cell + f'.{dataset.res_name}', 'cluster'] for cell in hic.obs_names])
        hic.obs['depth'] = np.array([dataset.reference.loc[cell + f'.{dataset.res_name}', 'depth'] for cell in hic.obs_names])
        hic.obs['batch'] = np.array([dataset.reference.loc[cell + f'.{dataset.res_name}', 'batch'] for cell in hic.obs_names])

        hic.obs['dataset'] = 'train'
        hic.write(base_hic_path, compression="gzip")
    else:
        hic = ad.read_h5ad(base_hic_path)
    print('Analyzing Hi-C data...')
    top_loop_mask = np.ones(hic.shape[1], dtype=bool)
    if n_distal_interactions is not None:
        if hic.shape[1] > n_distal_interactions:
            print('Identifying highly variable interactions...')
            sc.pp.highly_variable_genes(hic, n_top_genes=n_distal_interactions, flavor="seurat_v3", span=1)
            top_loop_mask = hic.var['highly_variable'].values
        
    print('Filtering distal interactions...')
    gene_mask_count, _ = sc.pp.filter_genes(hic, min_counts=min_count, inplace=False)
    gene_mask_cells, _ = sc.pp.filter_genes(hic, min_cells=min_count, inplace=False)
    gene_mask = gene_mask_count & gene_mask_cells & top_loop_mask
    non_distal_mask = np.array(hic.var_names.map(lambda s: s[-2] != '-' and s[-3] != '-').values, dtype=bool)
    gene_mask = gene_mask | non_distal_mask # create non distal mask
    # then figure out which root non distal features need to be kept
    keep_distal_mask = hic.var_names.map(lambda s: s[-2] == '-' or s[-3] == '-')
    distal_source = hic.var_names[keep_distal_mask].map(lambda s: s.rsplit('-', 1)[0]).values
    hic = hic[:, gene_mask].copy()  # first filter out all distal features
    # then filter out the root non distal features that are not needed
    keep_non_distal_mask = hic.var_names.isin(distal_source)
    # now filter out the non distal features keeping the ones that are needed
    gene_mask_count, _ = sc.pp.filter_genes(hic, min_counts=min_count, inplace=False)
    gene_mask_cells, _ = sc.pp.filter_genes(hic, min_cells=min_count, inplace=False)
    gene_mask = gene_mask_count & gene_mask_cells
    gene_mask = gene_mask | keep_non_distal_mask
    hic = hic[:, gene_mask].copy()
    print(hic)
    sc.pp.normalize_total(hic)
    sc.pp.log1p(hic)
    sc.pp.scale(hic)
    sc.tl.pca(hic, n_comps=min(100, hic.shape[0]), svd_solver="auto")
    sc.pp.neighbors(hic, n_pcs=min(100, hic.shape[0]), metric="cosine")
    sc.tl.umap(hic)
    fig = sc.pl.umap(hic, color=["celltype", "batch"], return_fig=True)
    fig.savefig(f'{plot_dir}/hic_umap_{resolution}.png')
    plt.close() 

    if atac_file is not None:
        atac, atac_peaks = load_atac_modality(
            atac_file, plot_dir, n_atac_peaks, hic.shape[1]
        )
    if methyl_file is not None:
        methyl, methyl_genes = load_methyl_modality(
            methyl_file, rna, gtf_file, n_genes, plot_dir
        )

    genes = scglue.genomics.Bed(rna.var.assign(name=rna.var_names))
    diagonal_mask = hic.var_names.map(lambda s: s[-2] != '-' and s[-3] != '-')
    diagonal_anchors = hic.var[diagonal_mask]
    peaks = scglue.genomics.Bed(diagonal_anchors.assign(name=hic.var_names[diagonal_mask]))
    tss = genes.strand_specific_start_site()
    promoters = tss.expand(2000, 0)
    if atac_file is not None:
        atac_peaks = scglue.genomics.Bed(atac_peaks.assign(name=atac_peaks['name']))
    if methyl_file is not None: 
        methyl_genes = scglue.genomics.Bed(methyl_genes.assign(name=methyl_genes['name']))
        methyl_tss = methyl_genes.strand_specific_start_site()
        methyl_promoters = methyl_tss.expand(2000, 0)

    

    frags['peak_name'] = frags.apply(lambda row: f"{row['chrom']}:{row['chromStart']}-{row['chromEnd']}", axis=1)
    sign_dict = {}
    def sign_map(interval):
        if interval in sign_dict.keys():
            print(interval, sign_dict[interval])
            return sign_dict[interval]
        else:
            return 1

    overlap_graph = scglue.genomics.window_graph(
        promoters, peaks, 0,
        attr_fn=lambda l, r, d: {
            "weight": 1.0,
            "type": "overlap",
            "sign": sign_map(r.name)
        }
    )
    overlap_graph = nx.DiGraph(overlap_graph)
    print('Overlap #edges:', overlap_graph.number_of_edges())
    if atac_file is not None:
        atac_overlap_graph = scglue.genomics.window_graph(
            promoters, atac_peaks, 0,
            attr_fn=lambda l, r, d: {
                "weight": 1.0,
                "type": "overlap",
                "sign": sign_map(r.name)
            }
        )
        atac_hic_overlap_graph = scglue.genomics.window_graph(
            atac_peaks, peaks, 0,
            attr_fn=lambda l, r, d: {
                "weight": 1.0,
                "type": "overlap",
                "sign": sign_map(r.name)
            }
        )
        print('ATAC graph:', atac_overlap_graph)
        print('ATAC-HiC graph:', atac_hic_overlap_graph)
    if methyl_file is not None:
        methyl_overlap_graph = scglue.genomics.window_graph(
            promoters, methyl_promoters, 0,
            attr_fn=lambda l, r, d: {
                "weight": 1.0,
                "type": "overlap",
                "sign": -1
            }
        )
        methyl_hic_overlap_graph = scglue.genomics.window_graph(
            methyl_promoters, peaks, 0,
            attr_fn=lambda l, r, d: {
                "weight": 1.0,
                "type": "overlap",
                "sign": -1
            }
        )
        print('Methyl graph:', methyl_overlap_graph)
        print('Methyl-HiC graph:', methyl_hic_overlap_graph)

    dist_graph = scglue.genomics.window_graph(
        promoters, peaks, 150000,
        attr_fn=lambda l, r, d: {
            "dist": abs(d) / 2e6,
            "weight": scglue.genomics.dist_power_decay(abs(d)),
            "type": "dist",
            "sign": sign_map(r.name)
        }
    )
    dist_graph = nx.DiGraph(dist_graph)
    print('Distance #edges:', dist_graph.number_of_edges())

    bait_oe = loops[['bin1_id', 'bin2_id', 'rank']]
    if loops_offset != 0:
        bait_oe['bin2_id'] -= loops_offset
    
    sorted_frags = frags.sort_values(by=['chrom', 'chromStart']).copy()
    # Keep chromosome-adjacent bins connected even when they fall below the loop cutoff.
    seq_loops = sorted_frags[['name', 'peak_name', 'chromStart', 'chromEnd', 'chrom']].copy()
    seq_loops['bin2_id'] = seq_loops.groupby('chrom')['name'].shift(-1)
    seq_loops['peak2'] = seq_loops.groupby('chrom')['peak_name'].shift(-1)
    seq_loops['start2'] = seq_loops.groupby('chrom')['chromStart'].shift(-1)
    seq_loops['end2'] = seq_loops.groupby('chrom')['chromEnd'].shift(-1)
    seq_loops = seq_loops.dropna(subset=['bin2_id']).rename(columns={
        'name': 'bin1_id',
        'peak_name': 'peak1',
        'chromStart': 'start1',
        'chromEnd': 'end1'
    })
    seq_loops['bin1_id'] = seq_loops['bin1_id'].astype(str)
    seq_loops['bin2_id'] = seq_loops['bin2_id'].astype(str)
    seq_loops['mid1'] = (seq_loops['end1'] - seq_loops['start1']).abs()
    seq_loops['mid2'] = (seq_loops['end2'] - seq_loops['start2']).abs()
    seq_loops['dist'] = (seq_loops['mid1'] - seq_loops['mid2']).abs() / 2e6
    seq_loops['peak1'] = seq_loops['peak1'].astype(str)
    seq_loops['peak2'] = seq_loops['peak2'].astype(str)
    seq_loops['rank'] = 1.0
    seq_loops = seq_loops[['peak1', 'peak2', 'rank', 'dist']]
    
    bait_oe['bin1_id'] = bait_oe['bin1_id'].astype(str)
    bait_oe['bin2_id'] = bait_oe['bin2_id'].astype(str)
    bait_oe = bait_oe.dropna().reset_index(drop=True)

    peak_map = frags.set_index('name')['peak_name'].to_dict()
    start_map = frags.set_index('name')['chromStart'].to_dict()
    end_map = frags.set_index('name')['chromEnd'].to_dict()

    #peaks = frags.set_index('peak_name')
    bait_oe['peak1'] = bait_oe['bin1_id'].map(peak_map)
    bait_oe['peak2'] = bait_oe['bin2_id'].map(peak_map)
    print(bait_oe)
    bait_oe = bait_oe.dropna().reset_index(drop=True)
    bait_oe['start1'] = bait_oe['bin1_id'].map(start_map).astype(int)
    bait_oe['start2'] = bait_oe['bin2_id'].map(start_map).astype(int)
    bait_oe['end1'] = bait_oe['bin1_id'].map(end_map).astype(int)
    bait_oe['end2'] = bait_oe['bin2_id'].map(end_map).astype(int)
    bait_oe['mid1'] = (bait_oe['end1'] - bait_oe['start1']).abs()
    bait_oe['mid2'] = (bait_oe['end2'] - bait_oe['start2']).abs()
    bait_oe['dist'] = (bait_oe['mid1'] - bait_oe['mid2']).abs() / 2e6
    bait_oe['peak1'] = bait_oe['peak1'].astype(str)
    bait_oe['peak2'] = bait_oe['peak2'].astype(str)

    bait_oe = bait_oe[['peak1', 'peak2', 'rank', 'dist']]
    existing_hic_edges = set(zip(bait_oe['peak1'], bait_oe['peak2']))
    seq_loops = seq_loops.loc[
        [edge not in existing_hic_edges for edge in zip(seq_loops['peak1'], seq_loops['peak2'])]
    ].copy()
    bait_oe = pd.concat([bait_oe, seq_loops], ignore_index=True)

    frags.index = frags['peak_name']
    frags = scglue.genomics.Bed(frags)
    bait_oe.rename(columns={'rank': 'weight'}, inplace=True)
    pchic_graph = nx.from_pandas_edgelist(bait_oe, source="peak1", target="peak2", edge_attr=True, create_using=nx.DiGraph)
    

    nx.set_edge_attributes(pchic_graph, "hic", "type")
    nx.set_edge_attributes(pchic_graph, 1, "sign")
    print('Hi-C #edges:', pchic_graph.number_of_edges())

    gene_bait = scglue.genomics.window_graph(promoters, frags, 1000)

    chrom_attr = {}
    pos_attr = {}
    type_attr = {}

    rna.var["in_hic"] = biadjacency_matrix(gene_bait, genes.index).sum(axis=1) != 0
    print('Genes in Hi-C', rna.var["in_hic"].sum())

    o_prior = overlap_graph.copy()

    hvg_reachable = scglue.graph.reachable_vertices(o_prior, rna.var.query("highly_variable").index)

    hic.var["o_highly_variable"] = [item in hvg_reachable for item in hic.var_names]
    print('Overlap variable Hi-C features', hic.var["o_highly_variable"].sum())

    o_prior = scglue.graph.compose_multigraph(o_prior, o_prior.reverse())
    for item in itertools.chain(hic.var_names, rna.var_names):
        try:
            if item[-2] == '-' and item.startswith('chr'):
                continue
        except Exception as e:
            pass
        o_prior.add_edge(item, item, weight=1.0, type="self-loop", sign=1)
    nx.set_edge_attributes(o_prior, 1, "sign")
    nx.set_edge_attributes(o_prior, 0.0, "dist")
    o_prior = o_prior.subgraph(hvg_reachable)
    if atac_file is not None:
        atac_o_prior = atac_overlap_graph.copy()
        atac_hic_o_prior = atac_hic_overlap_graph.copy()
        # first limit to the highly variable ATAC peaks
        #atac_reachable = scglue.graph.reachable_vertices(atac_hic_o_prior, atac.var.query("highly_variable").index)
        #atac_hic_o_prior = atac_hic_o_prior.subgraph(atac_reachable)

        atac_o_prior = scglue.graph.compose_multigraph(atac_o_prior, atac_o_prior.reverse())
        atac_hic_o_prior = scglue.graph.compose_multigraph(atac_hic_o_prior, atac_hic_o_prior.reverse())
        atac_hic_o_prior = scglue.graph.compose_multigraph(atac_o_prior, atac_hic_o_prior)
        #atac_hic_o_prior = scglue.graph.compose_multigraph(atac_hic_o_prior, pchic_graph)
        # for item in itertools.chain(atac.var_names):
        #     atac_o_prior.add_edge(item, item, weight=1.0, type="self-loop", sign=1)
        #     atac_hic_o_prior.add_edge(item, item, weight=1.0, type="self-loop", sign=1)
        #atac_reachable = scglue.graph.reachable_vertices(atac_o_prior, rna.var.query("highly_variable").index)
        #atac_reachable_hvgs = scglue.graph.reachable_vertices(atac_hic_o_prior, rna.var.query("highly_variable").index)
        
        #atac.var["highly_variable"] = [item in atac_reachable for item in atac.var_names]
        #atac_o_prior = atac_o_prior.subgraph(atac_reachable)
        #atac_o_prior = atac_hic_o_prior.subgraph(atac_reachable_hvgs)
        atac_o_prior = atac_hic_o_prior
        # remove duplicate edges
        atac_o_prior = scglue.graph.compose_multigraph(atac_o_prior, atac_o_prior.reverse())
        for item in itertools.chain(atac.var_names):
            atac_o_prior.add_edge(item, item, weight=1.0, type="self-loop", sign=1)
        # remove nodes that are note connected to either a gene or a Hi-C node
        remove_nodes = []
        for node in tqdm(atac_o_prior.nodes):
            if node in atac.var_names:
                remove = True
                for neighbor in atac_o_prior.neighbors(node):
                    if neighbor in rna.var_names or neighbor in hic.var_names:
                        remove = False
                        break
                if remove:
                    remove_nodes.append(node)
        atac_o_prior.remove_nodes_from(remove_nodes)
        print('ATAC overlap graph:', atac_o_prior)
    if methyl_file is not None:
        methyl_o_prior = methyl_overlap_graph.copy()
        methyl_hic_o_prior = methyl_hic_overlap_graph.copy()
        # first limit to the highly variable methyl peaks
        #methyl_reachable = scglue.graph.reachable_vertices(methyl_hic_o_prior, methyl.var.query("highly_variable").index)
        #methyl_hic_o_prior = methyl_hic_o_prior.subgraph(methyl_reachable)

        methyl_o_prior = scglue.graph.compose_multigraph(methyl_o_prior, methyl_o_prior.reverse())
        methyl_hic_o_prior = scglue.graph.compose_multigraph(methyl_hic_o_prior, methyl_hic_o_prior.reverse())
        methyl_hic_o_prior = scglue.graph.compose_multigraph(methyl_o_prior, methyl_hic_o_prior)
        #methyl_hic_o_prior = scglue.graph.compose_multigraph(methyl_hic_o_prior, pchic_graph)
        # for item in itertools.chain(methyl.var_names):
        #     methyl_o_prior.add_edge(item, item, weight=1.0, type="self-loop", sign=1)
        #     methyl_hic_o_prior.add_edge(item, item, weight=1.0, type="self-loop", sign=1)
        #methyl_reachable = scglue.graph.reachable_vertices(methyl_o_prior, rna.var.query("highly_variable").index)
        #methyl_reachable_hvgs = scglue.graph.reachable_vertices(methyl_hic_o_prior, rna.var.query("highly_variable").index)
        
        #methyl.var["highly_variable"] = [item in methyl_reachable for item in methyl.var_names]
        #methyl_o_prior = methyl_o_prior.subgraph(methyl_reachable)
        #methyl_o_prior = methyl_hic_o_prior.subgraph(methyl_reachable_hvgs)
        methyl_o_prior = methyl_hic_o_prior
        # remove duplicate edges
        methyl_o_prior = scglue.graph.compose_multigraph(methyl_o_prior, methyl_o_prior.reverse())
        for item in itertools.chain(methyl.var_names):
            methyl_o_prior.add_edge(item, item, weight=1.0, type="self-loop", sign=1)
        # remove nodes that are note connected to either a gene or a Hi-C node
        remove_nodes = []
        for node in tqdm(methyl_o_prior.nodes):
            if node in methyl.var_names:
                remove = True
                for neighbor in methyl_o_prior.neighbors(node):
                    if neighbor in rna.var_names or neighbor in hic.var_names:
                        remove = False
                        break
                if remove:
                    remove_nodes.append(node)
        methyl_o_prior.remove_nodes_from(remove_nodes)
        print('Methyl overlap graph:', methyl_o_prior)

    d_prior = dist_graph.copy()

    # any genes which are connected to a highly variable peak are also highly variable
    if atac_file is not None:
        rna_hvg_set = set(rna.var.query("highly_variable").index)
        print('Initial highly variable genes:', len(rna_hvg_set))
        atac_hvg_set = set(atac.var.query("highly_variable").index)
        reachable_from_hv_peaks = scglue.graph.reachable_vertices(atac_o_prior, atac_hvg_set)
        hv_genes_from_peaks = {v for v in reachable_from_hv_peaks if not v.startswith("chr")}
        reachable_from_hv_genes = scglue.graph.reachable_vertices(atac_o_prior, rna_hvg_set)
        hv_peaks_from_genes = {v for v in reachable_from_hv_genes if v.startswith("chr")}
        final_hv_genes = rna_hvg_set.union(hv_genes_from_peaks)
        final_hv_peaks = atac_hvg_set.union(hv_peaks_from_genes)
        rna.var["highly_variable"] = rna.var_names.isin(final_hv_genes)
        atac.var["highly_variable"] = atac.var_names.isin(final_hv_peaks)
        print('Final highly variable genes after adding ATAC:', rna.var["highly_variable"].sum())
        print('Final highly variable ATAC peaks after adding genes:', atac.var["highly_variable"].sum())
    if methyl_file is not None:
        rna_hvg_set = set(rna.var.query("highly_variable").index)
        methyl_hvg_set = set(methyl.var.query("highly_variable").index)
        reachable_from_hv_peaks = scglue.graph.reachable_vertices(methyl_o_prior, methyl_hvg_set)
        hv_genes_from_peaks = {v for v in reachable_from_hv_peaks if not v.startswith("chr")}
        reachable_from_hv_genes = scglue.graph.reachable_vertices(methyl_o_prior, rna_hvg_set)
        hv_peaks_from_genes = {v for v in reachable_from_hv_genes if v.startswith("chr")}
        final_hv_genes = rna_hvg_set.union(hv_genes_from_peaks)
        final_hv_peaks = methyl_hvg_set.union(hv_peaks_from_genes)
        rna.var["highly_variable"] = rna.var_names.isin(final_hv_genes)
        methyl.var["highly_variable"] = methyl.var_names.isin(final_hv_peaks)
        print('Final highly variable genes after adding Methyl:', rna.var["highly_variable"].sum())
        print('Final highly variable Methyl genes after adding genes:', methyl.var["highly_variable"].sum())

    if atac_file is not None:
        atac_hvg_reachable = scglue.graph.reachable_vertices(atac_o_prior, atac.var.query("highly_variable").index)
        atac_o_prior = atac_o_prior.subgraph(atac_hvg_reachable)
        print(atac_o_prior)
    if methyl_file is not None:
        methyl_hvg_reachable = scglue.graph.reachable_vertices(methyl_o_prior, methyl.var.query("highly_variable").index)
        methyl_o_prior = methyl_o_prior.subgraph(methyl_hvg_reachable)

    hvg_reachable = scglue.graph.reachable_vertices(d_prior, rna.var.query("highly_variable").index)

    hic.var["d_highly_variable"] = [item in hvg_reachable for item in hic.var_names]
    print('Distance variable Hi-C features', hic.var["d_highly_variable"].sum())

    d_prior = scglue.graph.compose_multigraph(d_prior, d_prior.reverse())
    for item in itertools.chain(hic.var_names, rna.var_names):
        try:
            if item[-2] == '-' and item.startswith('chr'):
                continue
        except Exception as e:
            pass
        d_prior.add_edge(item, item, weight=1.0, type="self-loop", sign=1)

    nx.set_edge_attributes(d_prior, 1, "sign")
    d_prior = d_prior.subgraph(hvg_reachable)
    print(pchic_graph)
    pchic_graph = scglue.graph.compose_multigraph(pchic_graph, pchic_graph.reverse())
    
    print('Composing overlap graph with Hi-C graph...')
    dcq_prior = scglue.graph.compose_multigraph(o_prior, pchic_graph)
    nx.set_edge_attributes(dcq_prior, 1, "sign")
    if atac_file is not None:
        print('Composing ATAC overlap graph with Hi-C graph...')
        dcq_prior = scglue.graph.compose_multigraph(dcq_prior, atac_o_prior)
    if methyl_file is not None:
        print('Composing Methyl overlap graph with Hi-C graph...')
        dcq_prior = scglue.graph.compose_multigraph(dcq_prior, methyl_o_prior)

    hvg_reachable = scglue.graph.reachable_vertices(dcq_prior, rna.var.query("highly_variable").index)
    hic.var["dcq_highly_variable"] = [item in hvg_reachable for item in hic.var_names]
    if atac_file is not None:
        atac_reachable = scglue.graph.reachable_vertices(dcq_prior, rna.var.query("highly_variable").index)
        atac.var["highly_variable"] = [item in atac_reachable for item in atac.var_names]
        print('ATAC variable Hi-C features', atac.var["highly_variable"].sum())
    if methyl_file is not None:
        methyl_reachable = scglue.graph.reachable_vertices(dcq_prior, rna.var.query("highly_variable").index)
        methyl.var["highly_variable"] = [item in methyl_reachable for item in methyl.var_names]
        print('Methyl variable Hi-C features', methyl.var["highly_variable"].sum())
    keep_distal_mask = hic.var_names.map(lambda s: s[-2] == '-' or s[-3] == '-')
    # set distal entries as highly variable too
    hic.var["dcq_highly_variable"] = hic.var["dcq_highly_variable"] | keep_distal_mask  
    # and keep all roots of distal entries if they aren't already 
    distal_source = hic.var_names[keep_distal_mask].map(lambda s: s.rsplit('-', 1)[0]).values
    keep_non_distal_mask = hic.var_names.isin(distal_source)
    hic.var["dcq_highly_variable"] = hic.var["dcq_highly_variable"] | keep_non_distal_mask
    print('Full Hi-C prior variable features:', hic.var["dcq_highly_variable"].sum())

    for item in dcq_prior.nodes:
        if dcq_prior.has_edge(item, item):
            continue
        else:
            dcq_prior.add_edge(item, item, weight=1.0, type="self-loop", sign=1)

    chrom_attr = {}
    strata_attr = {}
    pos_attr = {}
    type_attr = {}
    for n in tqdm(dcq_prior.nodes):
        if n in genes.index:
            type_attr[n] = 'RNA'
            row = genes.loc[n]
            strata_attr[n] = 0
        elif n in frags.index:
            type_attr[n] = 'Hi-C'
            row = frags.loc[n]
            strata_attr[n] = 0.0
        else:
            if atac_file is not None and n in atac.var_names:
                type_attr[n] = 'ATAC'
                row = atac_peaks.loc[n]
                strata_attr[n] = 0.0
            elif methyl_file is not None and n in methyl.var_names:
                type_attr[n] = 'Methyl'
                row = methyl_genes.loc[n]
                strata_attr[n] = 0.0
            else:
                type_attr[n] = 'Hi-C'
                chr_split = str(n).split(':')
                row = {}
                row['chrom'] = str(chr_split[0])
                try:
                    pos = chr_split[1].split('-')
                    row['chromStart'] = int(pos[0])
                except Exception as e:
                    print(n, chr_split, pos, e)
                    row['chromStart'] = 0
                try:
                    strata_attr[n] = int(str(n).split('-')[-1])
                except Exception as e:
                    print(e)
                    strata_attr[n] = 0
        chrom_attr[n] = str(row['chrom'])
        pos_attr[n] = int(row['chromStart'])

    edge_ids = {}
    for edge_i, e in enumerate(dcq_prior.edges):
        edge_ids[e] = edge_i
    nx.set_edge_attributes(dcq_prior, edge_ids, "edge_id")

    if use_ice:
        suffix = f'ice_{loop_q}'
    elif use_raw_pseudobulk:
        suffix = f'raw_{loop_q}'
    else:
        suffix = f'deeploop_{loop_q}'
    suffix += f'_2d_{n_strata}'
    print(suffix)
    os.makedirs(os.path.join(out_dir, 'rna'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'hic'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'atac'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'methyl'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'graphs'), exist_ok=True)

    hic.write(f"{out_dir}/hic/hic_{resolution}_{suffix}.h5ad", compression="gzip")
    rna.write(f"{out_dir}/rna/rna_{resolution}_{suffix}.h5ad", compression="gzip")
    if atac_file is not None:
        atac.write(f"{out_dir}/atac/atac_{resolution}_{suffix}.h5ad", compression="gzip")
    if methyl_file is not None:
        methyl.write(f"{out_dir}/methyl/methyl_{resolution}_{suffix}.h5ad", compression="gzip")

    nx.set_node_attributes(dcq_prior, chrom_attr, "chrom")
    nx.set_node_attributes(dcq_prior, pos_attr, "chrom_pos")
    nx.set_node_attributes(dcq_prior, type_attr, "feature_type")
    nx.write_graphml(dcq_prior, f"{out_dir}/graphs/dcq_prior_{resolution}_{suffix}.graphml.gz", edge_id_from_attribute='edge_id', named_key_ids=True)