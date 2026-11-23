#!/usr/bin/env bash
# Collects the inputs of the main HiGLUE benchmark (benchmark_main.sh) in one
# tree of symbolic links, with the same file names for every dataset:
#
#   datasets/<dataset>/rna.h5ad          RNA reference (raw counts, obs[celltype])
#   datasets/<dataset>/genes.gtf         gene annotation of the Hi-C genome assembly
#   datasets/<dataset>/hic_ref.tsv       scHi-C metadata (cell, depth, batch, cluster)
#   datasets/<dataset>/hic_<res>.scool   single-cell coolers at each available resolution
#
# Shared files (GTFs, RNA references used by several datasets) live in
# datasets/_shared/ and are linked from the dataset directories with relative
# links, so that sync_to_cluster.sh copies them only once.
#
# Usage:
#   datasets/link_datasets.sh            # (re)create all links, report missing sources
#   datasets/link_datasets.sh --check    # only report missing sources
#
# This file is the single place where source paths are recorded; edit the
# table below when a source moves.

set -uo pipefail
cd "$(dirname "$0")"

CHECK_ONLY=0
[[ ${1:-} == --check ]] && CHECK_ONLY=1

J1=/mnt/jinstore/JinLab01/dmp131
J2=/mnt/jinstore/JinLab02/dmp131
J4=/mnt/jinstore/JinLab04/dmp131

# link | source ("@path" = relative link to another file of this tree)
LINKS=(
    # ---- shared
    "_shared/genomes/hg19.gtf|$J1/scRNA/hg19_genes.gtf"
    "_shared/genomes/hg38.gtf|$J1/scRNA/hg38_genes.gtf"
    "_shared/genomes/mm10.gtf|$J2/scHi-C/HiRes_mouse_brain/ref/gencode.vM25.annotation.gtf"
    "_shared/rna/pancdb_islet.h5ad|$J2/scRNA/pancdb/islet_processed.h5ad"  # PancDB / HPAP islets
    "_shared/rna/inhouse_dropseq_islet.h5ad|$J2/scRNA/in_house/islet_processed.h5ad"  # in-house islet Drop-seq (counts in layers)

    # ---- human prefrontal cortex: sn-m3C-seq (Lee et al. 2019) + snRNA-seq (Velmeshev et al. 2019), hg19
    "pfc/rna.h5ad|$J1/scRNA/brain_autism.h5ad"
    "pfc/genes.gtf|@_shared/genomes/hg19.gtf"
    "pfc/hic_ref.tsv|$J2/HiGLUE/data/pfc_neurons_ref"
    # Built by datasets/build_pfc_scools.py from the fragment-level loops (uniform bins,
    # cis pairs >= 2.5kb, trans kept). Earlier PFC scools (scools/pfc_500kb.scool,
    # PFC_rebin) went through fragdata_to_anchordata.pl, which drops every same-bin
    # contact. In runs 181218 and 190305 the archived cis loops lack chr1, chr2 and
    # chr11-22 in every version (lost upstream); the depth filter keeps 19 of those cells.
    # pfc_frag_<res>_chr3-10X.scool: only the chromosomes complete in every run
    "pfc/hic_50kb.scool|$J1/scHi-C/scools/pfc_frag_50kb.scool"
    "pfc/hic_100kb.scool|$J1/scHi-C/scools/pfc_frag_100kb.scool"
    "pfc/hic_200kb.scool|$J1/scHi-C/scools/pfc_frag_200kb.scool"
    "pfc/hic_500kb.scool|$J1/scHi-C/scools/pfc_frag_500kb.scool"
    # the same PFC data with all 14 cell types (pfc: neurons only), RNA without the neuron subset (higlue.py)
    "pfc_full/rna.h5ad|$J1/scRNA/brain_autism.h5ad"
    "pfc_full/genes.gtf|@_shared/genomes/hg19.gtf"
    "pfc_full/hic_ref.tsv|$J1/scHi-C/pfc_ref"
    "pfc_full/hic_50kb.scool|$J1/scHi-C/scools/pfc_frag_50kb.scool"
    "pfc_full/hic_100kb.scool|$J1/scHi-C/scools/pfc_frag_100kb.scool"
    "pfc_full/hic_500kb.scool|$J1/scHi-C/scools/pfc_frag_500kb.scool"

    # ---- human brain atlas neurons: snm3C-seq (Tian et al. 2023) + snRNA-seq (Siletti et al. 2023), hg38
    "human_brain/rna.h5ad|$J2/scRNA/siletti_human_brain_filtered_50k_preprocessed.h5ad"
    "human_brain/genes.gtf|@_shared/genomes/hg38.gtf"
    "human_brain/hic_ref.tsv|$J2/scHi-C/human_brain_snm3cseq_2023/human_brain_neurons_ref"
    # coarsened_new, not the earlier *_remap_* scools: those displace ~17% of each cell's
    # contacts to long range, and in a subset of cells most chr3/chr4 contacts, which
    # splits every cell type in two. Both files hold all atlas cells; the reference
    # selects the neurons
    "human_brain/hic_100kb.scool|$J2/scHi-C/scools/human_brain_100kb_coarsened_new.scool"
    "human_brain/hic_500kb.scool|$J2/scHi-C/scools/human_brain_500kb_coarsened_new.scool"

    # ---- mouse brain: HiRES, paired scHi-C + scRNA-seq (Liu et al. 2023), mm10
    "hires_brain/rna.h5ad|$J2/scHi-C/HiRes_mouse_brain/hires_brain_rna.h5ad"
    "hires_brain/genes.gtf|@_shared/genomes/mm10.gtf"
    "hires_brain/hic_ref.tsv|$J2/scHi-C/HiRes_mouse_brain/hires_brain_ref"
    "hires_brain/hic_5kb.scool|$J2/scHi-C/HiRes_mouse_brain/data/scools/hires_brain_5kb.scool"
    "hires_brain/hic_100kb.scool|$J2/scHi-C/HiRes_mouse_brain/data/scools/hires_brain_100kb.scool"
    "hires_brain/hic_200kb.scool|$J2/scHi-C/HiRes_mouse_brain/data/scools/hires_brain_200kb.scool"
    "hires_brain/hic_500kb.scool|$J2/scHi-C/HiRes_mouse_brain/data/scools/hires_brain_500kb.scool"

    # ---- mouse cortex: Droplet Hi-C paired multiome (Chang et al. 2025), mm10
    "droplet_cortex/rna.h5ad|$J2/scHi-C/droplet_cortex/droplet_mouse_cortex_rna_hic_barcodes.h5ad"  # RNA renamed to the Hi-C barcodes (replace_barcodes.py): pairs known, FOSCTTM
    "droplet_cortex/genes.gtf|@_shared/genomes/mm10.gtf"
    "droplet_cortex/hic_ref.tsv|$J2/scHi-C/droplet_cortex/droplet_mouse_cortex_ref"
    "droplet_cortex/hic_5kb.scool|$J2/scHi-C/droplet_cortex/data/scools/droplet_mouse_cortex_5kb.scool"
    "droplet_cortex/hic_100kb.scool|$J2/scHi-C/droplet_cortex/data/scools/droplet_mouse_cortex_100kb.scool"
    "droplet_cortex/hic_200kb.scool|$J2/scHi-C/droplet_cortex/data/scools/droplet_mouse_cortex_200kb.scool"
    "droplet_cortex/hic_500kb.scool|$J2/scHi-C/droplet_cortex/data/scools/droplet_mouse_cortex_500kb.scool"
    # Droplet Hi-C cortex with an unpaired SNARE-seq RNA + ATAC co-assay (Chen 2019; triple integration, Figure 4)
    "droplet_snare/rna.h5ad|$J2/scRNA/chen_SNAREseq/Chen-2019-RNA.h5ad"
    "droplet_snare/atac.h5ad|$J2/scRNA/chen_SNAREseq/Chen-2019-ATAC.h5ad"
    "droplet_snare/genes.gtf|@_shared/genomes/mm10.gtf"
    "droplet_snare/hic_ref.tsv|$J2/scHi-C/droplet_cortex/droplet_mouse_cortex_ref"
    "droplet_snare/hic_5kb.scool|$J2/scHi-C/droplet_cortex/data/scools/droplet_mouse_cortex_5kb.scool"
    "droplet_snare/hic_500kb.scool|$J2/scHi-C/droplet_cortex/data/scools/droplet_mouse_cortex_500kb.scool"
    # ---- GAGE-seq (Zhou et al. 2024, GSE238001), paired scHi-C + scRNA of the same cells, authors' annotation
    # (GitHub ma-compbio/GAGE-seq meta_info, Oct 2025); built by datasets/build_gageseq.py; cell names shared by RNA and Hi-C
    "gageseq_cortex/rna.h5ad|$J4/scHi-C/GAGEseq/cortex/rna.h5ad"
    "gageseq_cortex/genes.gtf|@_shared/genomes/mm10.gtf"
    "gageseq_cortex/hic_ref.tsv|$J4/scHi-C/GAGEseq/cortex/hic_ref.tsv"
    "gageseq_cortex/hic_5kb.scool|$J4/scHi-C/GAGEseq/cortex/hic_5kb.scool"
    "gageseq_cortex/hic_50kb.scool|$J4/scHi-C/GAGEseq/cortex/hic_50kb.scool"
    "gageseq_cortex/hic_100kb.scool|$J4/scHi-C/GAGEseq/cortex/hic_100kb.scool"
    "gageseq_cortex/hic_200kb.scool|$J4/scHi-C/GAGEseq/cortex/hic_200kb.scool"
    "gageseq_cortex/hic_500kb.scool|$J4/scHi-C/GAGEseq/cortex/hic_500kb.scool"
    "gageseq_bm/rna.h5ad|$J4/scHi-C/GAGEseq/bm/rna.h5ad"
    "gageseq_bm/genes.gtf|@_shared/genomes/hg38.gtf"
    "gageseq_bm/hic_ref.tsv|$J4/scHi-C/GAGEseq/bm/hic_ref.tsv"
    "gageseq_bm/hic_5kb.scool|$J4/scHi-C/GAGEseq/bm/hic_5kb.scool"
    "gageseq_bm/hic_50kb.scool|$J4/scHi-C/GAGEseq/bm/hic_50kb.scool"
    "gageseq_bm/hic_100kb.scool|$J4/scHi-C/GAGEseq/bm/hic_100kb.scool"
    "gageseq_bm/hic_200kb.scool|$J4/scHi-C/GAGEseq/bm/hic_200kb.scool"
    "gageseq_bm/hic_500kb.scool|$J4/scHi-C/GAGEseq/bm/hic_500kb.scool"

    # ---- mouse embryo E7.0-E11.5: HiRES, paired scHi-C + scRNA-seq (Liu et al. 2023, GSE223917), mm10
    # all 7,469 cells, built by datasets/build_hires_embryo_full.py from the GEO per-cell pairs
    "hires_embryo_full/rna.h5ad|$J2/scHi-C/HiRes_mouse_embryo/hires_embryo_rna.h5ad"
    "hires_embryo_full/genes.gtf|@_shared/genomes/mm10.gtf"
    "hires_embryo_full/hic_ref.tsv|$J4/scHi-C/HiRes_mouse_embryo_full/hires_embryo_full_ref"
    "hires_embryo_full/hic_5kb.scool|$J4/scHi-C/HiRes_mouse_embryo_full/scools/hires_embryo_full_5kb.scool"
    "hires_embryo_full/hic_100kb.scool|$J4/scHi-C/HiRes_mouse_embryo_full/scools/hires_embryo_full_100kb.scool"
    "hires_embryo_full/hic_200kb.scool|$J4/scHi-C/HiRes_mouse_embryo_full/scools/hires_embryo_full_200kb.scool"
    "hires_embryo_full/hic_500kb.scool|$J4/scHi-C/HiRes_mouse_embryo_full/scools/hires_embryo_full_500kb.scool"
    # the same cells without mitotic Hi-C cells (2-12 Mb / 25 kb-2 Mb cis ratio > 0.5) and without the RNA
    # "mitosis" type (embryo_benchmark/make_nomito.py); the paper's benchmark dataset
    "hires_embryo_full_nomito/rna.h5ad|$J4/scHi-C/HiRes_mouse_embryo_full/hires_embryo_full_nomito_rna.h5ad"
    "hires_embryo_full_nomito/genes.gtf|@_shared/genomes/mm10.gtf"
    "hires_embryo_full_nomito/hic_ref.tsv|$J4/scHi-C/HiRes_mouse_embryo_full/hires_embryo_full_nomito_ref"
    "hires_embryo_full_nomito/hic_5kb.scool|$J4/scHi-C/HiRes_mouse_embryo_full/scools/hires_embryo_full_5kb.scool"
    "hires_embryo_full_nomito/hic_100kb.scool|$J4/scHi-C/HiRes_mouse_embryo_full/scools/hires_embryo_full_100kb.scool"
    "hires_embryo_full_nomito/hic_200kb.scool|$J4/scHi-C/HiRes_mouse_embryo_full/scools/hires_embryo_full_200kb.scool"
    "hires_embryo_full_nomito/hic_500kb.scool|$J4/scHi-C/HiRes_mouse_embryo_full/scools/hires_embryo_full_500kb.scool"

    # ---- pancreatic islet cells of the snm3C-seq human body atlas (Zhou et al. 2026) + PancDB RNA, hg38
    # All 4,700 cells of the islet samples, built by datasets/build_islet_atlas_all.py,
    # with the atlas's stromal and immune cells labeled in the RNA vocabulary (PSC,
    # PSC_Quiescent, Endothelial, Macrophage). The earlier islet_cells_ref /
    # islet_atlas_<res>.scool kept only the six epithelial types (4,565 cells)
    "islet_atlas/rna.h5ad|@_shared/rna/pancdb_islet.h5ad"
    # gene-body mCH / mCG of the same snm3C-seq cells (zhoujt1994/HumanCellEpigenomeAtlas_sc_allc, islet_benchmark/methyl_gene_matrix.py)
    "islet_atlas/methyl.h5ad|$J4/scHi-C/islet/methylation_atlas/gene_level/islet_atlas_methyl_genes.h5ad"
    "islet_atlas/genes.gtf|@_shared/genomes/hg38.gtf"
    "islet_atlas/hic_ref.tsv|$J2/scHi-C/human_body_atlas/islet_all_cells_ref"
    "islet_atlas/hic_5kb.scool|$J2/scHi-C/human_body_atlas/data/scools/islet_atlas_all_5kb.scool"
    "islet_atlas/hic_100kb.scool|$J2/scHi-C/human_body_atlas/data/scools/islet_atlas_all_100kb.scool"
    "islet_atlas/hic_200kb.scool|$J2/scHi-C/human_body_atlas/data/scools/islet_atlas_all_200kb.scool"
    "islet_atlas/hic_500kb.scool|$J2/scHi-C/human_body_atlas/data/scools/islet_atlas_all_500kb.scool"

    # ---- in-house human islet Dip-C (FACS-sorted alpha/beta cells labeled, others Unknown) + PancDB RNA, hg19
    "islet_inhouse/rna.h5ad|@_shared/rna/pancdb_islet.h5ad"
    # in-house Drop-seq RNA as an alternative reference (round 5d: --rna_file datasets/islet_full/rna_dropseq.h5ad)
    "islet_full/rna_dropseq.h5ad|@_shared/rna/inhouse_dropseq_islet.h5ad"
    # islet atlas snm3C-seq gene-body methylation (unpaired with the in-house cells) on hg19 gene coordinates
    # (islet_benchmark/methyl_to_build.py)
    "islet_inhouse/methyl_atlas.h5ad|$J4/scHi-C/islet/methylation_atlas/gene_level/islet_atlas_methyl_genes_hg19.h5ad"
    "islet_inhouse/genes.gtf|@_shared/genomes/hg19.gtf"
    "islet_inhouse/hic_ref.tsv|$J4/scHi-C/islet/2026_islet_ref"
    "islet_inhouse/hic_5kb.scool|$J4/scHi-C/islet/data/scools/islet_2026_5kb.scool"
    "islet_inhouse/hic_50kb.scool|$J4/scHi-C/islet/data/scools/islet_2026_50kb.scool"
    "islet_inhouse/hic_100kb.scool|$J4/scHi-C/islet/data/scools/islet_2026_100kb.scool"
    "islet_inhouse/hic_200kb.scool|$J4/scHi-C/islet/data/scools/islet_2026_200kb.scool"
    "islet_inhouse/hic_500kb.scool|$J4/scHi-C/islet/data/scools/islet_2026_500kb.scool"
)

relpath() {  # relative path from directory $1 to file $2 (both within this tree)
    python3 -c 'import os, sys; print(os.path.relpath(sys.argv[2], sys.argv[1]))' "$1" "$2"
}

n_missing=0
for entry in "${LINKS[@]}"; do
    link=${entry%%|*}
    src=${entry#*|}
    if [[ $src == @* ]]; then
        target=${src#@}
        [[ -e $target ]] || {
            # the shared file may only be created later in this loop
            shared_src=$(printf '%s\n' "${LINKS[@]}" | grep "^${target}|" | cut -d'|' -f2)
            [[ -e $shared_src ]] || { echo "missing  $link -> $target"; n_missing=$((n_missing + 1)); continue; }
        }
        src=$(relpath "$(dirname "$link")" "$target")
    elif [[ ! -e $src ]]; then
        echo "missing  $link -> $src"
        n_missing=$((n_missing + 1))
        continue
    fi
    (( CHECK_ONLY )) && continue
    mkdir -p "$(dirname "$link")"
    ln -sfn "$src" "$link"
done

if (( ! CHECK_ONLY )); then
    for d in */; do
        [[ -d $d ]] || continue
        printf '%-16s %s\n' "${d%/}" "$(cd "$d" && ls | tr '\n' ' ')"
    done
fi
if (( n_missing )); then
    echo "$n_missing source(s) missing"
    exit 1
fi
echo "All sources present."
