# scGAD + Seurat CCA: BandNorm's runProjection (Shen, Zheng & Keles, Bioinformatics 2022; R/scGAD.R, lines 155-194)
# on the scGAD scores of the scHi-C cells and the scRNA-seq reference. Called by scgad.py (integrate_cca).
#
#   Rscript scgad_cca.R <workdir> <seed> [<k_filter> <k_weight> <npcs>]
#
# <workdir> holds (written by scgad.py): rna.mtx (genes x cells, raw counts), rna_genes.txt, rna_cells.txt,
# gad.mtx (genes x cells, raw GAD counts), gad_genes.txt, gad_cells.txt. Writes <workdir>/pca.csv: cell, domain,
# PC_1..PC_<npcs> of the integrated assay.
#
# As in runProjection: genes shared by both, the GAD scores are given as data and scale.data (no normalization;
# here the scores are those of scgad.gad_scores: counts per 10^4 per cell, z-scored per gene), the RNA is
# log-normalized (NormalizeData) and scaled (doNorm = TRUE for RNA, as for raw counts), 2,500 variable features per
# object (mean.var.plot), SelectIntegrationFeatures, FindIntegrationAnchors (CCA, k.filter = 50),
# IntegrateData (k.weight = 80), ScaleData, RunPCA (15 PCs). Only the RNA is scaled on its variable features
# (runProjection scales all genes, which does not change the anchor features and would not fit in memory for the
# 66k-cell references).
suppressPackageStartupMessages({library(Seurat); library(Matrix)})
options(Seurat.object.assay.version = "v3", future.globals.maxSize = 64 * 1024^3)
args <- commandArgs(trailingOnly = TRUE)
wd <- args[1]
seed <- as.integer(args[2])
k_filter <- if (length(args) >= 3) as.integer(args[3]) else 50L
k_weight <- if (length(args) >= 4) as.integer(args[4]) else 80L
npcs <- if (length(args) >= 5) as.integer(args[5]) else 15L
rna_mode <- if (length(args) >= 6) args[6] else "zscore"  # zscore: log-normalized RNA z-scored per gene, given as
# data and scale.data like the GAD scores (both on one scale, as runProjection with doNorm = FALSE for both);
# lognorm: NormalizeData only (doNorm = TRUE for the RNA)
set.seed(seed)

read_mat <- function(prefix) {
  m <- as(readMM(file.path(wd, paste0(prefix, ".mtx"))), "CsparseMatrix")
  rownames(m) <- readLines(file.path(wd, paste0(prefix, "_genes.txt")))
  colnames(m) <- readLines(file.path(wd, paste0(prefix, "_cells.txt")))
  m
}
rna <- read_mat("rna")
gad <- read_mat("gad")
common <- intersect(rownames(rna), rownames(gad))
rna <- rna[common, ]
gad <- gad[common, ]
colnames(rna) <- paste0("rna:", colnames(rna))
colnames(gad) <- paste0("hic:", colnames(gad))

# GAD scores: counts per 10^4 per cell, z-scored per gene; genes without variance dropped
tot <- Matrix::colSums(gad)
g <- as.matrix(gad %*% Diagonal(x = ifelse(tot > 0, 1e4 / pmax(tot, 1e-12), 0)))
mu <- rowMeans(g)
sdv <- sqrt(rowSums((g - mu)^2) / (ncol(g) - 1))
keep <- sdv > 0
z <- (g[keep, ] - mu[keep]) / sdv[keep]
dimnames(z) <- list(rownames(gad)[keep], colnames(gad))
rm(g)
cat(sprintf("[scGAD-CCA] %d RNA cells, %d Hi-C cells, %d shared genes (%d with GAD variance)\n",
            ncol(rna), ncol(z), length(common), nrow(z)))

s_gad <- CreateSeuratObject(counts = as(z, "CsparseMatrix"), min.cells = 0, min.features = 0)
s_gad <- SetAssayData(s_gad, assay = "RNA", slot = "data", new.data = as(z, "CsparseMatrix"))
s_gad <- SetAssayData(s_gad, assay = "RNA", slot = "scale.data", new.data = z)
s_gad$method <- "scGAD"
s_rna <- CreateSeuratObject(counts = rna[rownames(z), ], min.cells = 0, min.features = 0)
s_rna <- NormalizeData(s_rna, verbose = FALSE)
s_rna$method <- "scRNAseq"
lst <- lapply(list(s_gad, s_rna), function(x) FindVariableFeatures(x, selection.method = "mean.var.plot",
                                                                   nfeatures = 2500, verbose = FALSE))
if (rna_mode == "zscore") {
  vf <- VariableFeatures(lst[[2]])
  feats <- union(vf, SelectIntegrationFeatures(object.list = lst, verbose = FALSE))
  lst[[2]] <- subset(lst[[2]], features = feats)
  lst[[2]] <- ScaleData(lst[[2]], features = feats, verbose = FALSE)
  sd_rna <- GetAssayData(lst[[2]], slot = "scale.data")
  lst[[2]] <- SetAssayData(lst[[2]], slot = "data", new.data = as(sd_rna, "CsparseMatrix"))
  VariableFeatures(lst[[2]]) <- vf
} else {
  lst[[2]] <- ScaleData(lst[[2]], features = VariableFeatures(lst[[2]]), verbose = FALSE)
}
features <- SelectIntegrationFeatures(object.list = lst, verbose = FALSE)
cat(sprintf("[scGAD-CCA] %d integration features\n", length(features)))
anchors <- suppressWarnings(FindIntegrationAnchors(object.list = lst, anchor.features = features,
                                                   k.filter = k_filter, verbose = FALSE))
k_weight <- min(k_weight, ncol(z) - 1)
comb <- suppressWarnings(IntegrateData(anchorset = anchors, k.weight = k_weight, verbose = FALSE))
DefaultAssay(comb) <- "integrated"
comb <- suppressWarnings(ScaleData(comb, verbose = FALSE))
comb <- RunPCA(comb, npcs = npcs, verbose = FALSE, seed.use = seed)
pcs <- Embeddings(comb, "pca")
out <- data.frame(cell = sub("^(rna|hic):", "", rownames(pcs)), domain = sub(":.*$", "", rownames(pcs)), pcs,
                  check.names = FALSE)
write.csv(out, file.path(wd, "pca.csv"), row.names = FALSE)
cat(sprintf("[scGAD-CCA] wrote %d x %d PCs\n", nrow(pcs), ncol(pcs)))
