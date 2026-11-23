r"""
Training plugins
"""

import os
import time
import pathlib
import shutil
from typing import Iterable, List, Optional, Tuple

import wandb
import ignite
import ignite.contrib.handlers.tensorboard_logger as tb
import parse
import torch
import numpy as np
import scipy.sparse
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, accuracy_score, balanced_accuracy_score, f1_score, silhouette_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ..utils import config, logged
from ..num import normalize_edges
from .base import Trainer, TrainingPlugin
from .data import AnnDataset, DataLoader, GraphDataset
from ..data import transfer_labels
from . import viz

EPOCH_COMPLETED = ignite.engine.Events.EPOCH_COMPLETED
TERMINATE = ignite.engine.Events.TERMINATE
COMPLETED = ignite.engine.Events.COMPLETED


def get_closest_peak_to_gene(gene_name, rna, peaks):
    try:
        loc = rna.var.loc[gene_name]
    except KeyError:
        print('Could not find loci', gene_name)
        return None
    chrom = loc["chrom"]
    chromStart = loc["chromStart"]
    peaks['in_chr'] = peaks['chrom'] == chrom
    peaks['dist'] = peaks['chromStart'].apply(lambda s: abs(s - chromStart))
    peaks.loc[~peaks['in_chr'], 'dist'] = 1e9  # set distance to 1e9 if not in same chromosome
    return peaks['dist'].argmin()


class Tensorboard(TrainingPlugin):

    r"""
    Training logging via tensorboard
    """

    def attach(
            self, net: torch.nn.Module, trainer: Trainer,
            train_engine: ignite.engine.Engine,
            val_engine: ignite.engine.Engine,
            train_loader: Iterable,
            val_loader: Optional[Iterable],
            directory: pathlib.Path
    ) -> None:
        tb_directory = directory / "tensorboard"
        if tb_directory.exists():
            shutil.rmtree(tb_directory)

        tb_logger = tb.TensorboardLogger(
            log_dir=tb_directory,
            flush_secs=config.TENSORBOARD_FLUSH_SECS
        )
        tb_logger.attach(
            train_engine,
            log_handler=tb.OutputHandler(
                tag="train", metric_names=trainer.required_losses
            ), event_name=EPOCH_COMPLETED
        )
        if val_engine:
            tb_logger.attach(
                val_engine,
                log_handler=tb.OutputHandler(
                    tag="val", metric_names=trainer.required_losses
                ), event_name=EPOCH_COMPLETED
            )
        train_engine.add_event_handler(COMPLETED, tb_logger.close)


@logged
class EarlyStopping(TrainingPlugin):

    r"""
    Early stop model training when loss no longer decreases

    Parameters
    ----------
    monitor
        Loss to monitor
    patience
        Patience to stop early
    burnin
        Burn-in epochs to skip before initializing early stopping
    wait_n_lrs
        Wait n learning rate scheduling events before starting early stopping
    """

    def __init__(
            self, monitor: str, patience: int,
            burnin: int = 0, wait_n_lrs: int = 0
    ) -> None:
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.burnin = burnin
        self.wait_n_lrs = wait_n_lrs

    def attach(
            self, net: torch.nn.Module, trainer: Trainer,
            train_engine: ignite.engine.Engine,
            val_engine: ignite.engine.Engine,
            train_loader: Iterable,
            val_loader: Optional[Iterable],
            directory: pathlib.Path
    ) -> None:
        for item in directory.glob("checkpoint_*.pt"):
            item.unlink()

        score_engine = val_engine if val_engine else train_engine
        score_function = lambda engine: -score_engine.state.metrics[self.monitor]
        event_filter = (
            lambda engine, event: event > self.burnin and engine.state.n_lrs >= self.wait_n_lrs
        ) if self.wait_n_lrs else (
            lambda engine, event: event > self.burnin
        )
        event = EPOCH_COMPLETED(event_filter=event_filter)  # pylint: disable=not-callable
        train_engine.add_event_handler(
            event, ignite.handlers.Checkpoint(
                {"net": net, "trainer": trainer},
                ignite.handlers.DiskSaver(
                    directory, atomic=True, create_dir=True, require_empty=False
                ), score_function=score_function,
                filename_pattern="checkpoint_{global_step}.pt",
                n_saved=config.CHECKPOINT_SAVE_NUMBERS,
                global_step_transform=ignite.handlers.global_step_from_engine(train_engine)
            )
        )
        train_engine.add_event_handler(
            event, ignite.handlers.EarlyStopping(
                patience=self.patience,
                score_function=score_function,
                trainer=train_engine
            )
        )

        @train_engine.on(COMPLETED | TERMINATE)
        def _(engine):
            nan_flag = any(
                not bool(torch.isfinite(item).all())
                for item in (engine.state.output or {}).values()
            )
            ckpts = sorted([
                parse.parse("checkpoint_{epoch:d}.pt", item.name).named["epoch"]
                for item in directory.glob("checkpoint_*.pt")
            ], reverse=True)
            if ckpts and nan_flag and train_engine.state.epoch == ckpts[0]:
                self.logger.warning(
                    "The most recent checkpoint \"%d\" can be corrupted by NaNs, "
                    "will thus be discarded.", ckpts[0]
                )
                ckpts = ckpts[1:]
            if ckpts:
                self.logger.info("Restoring checkpoint \"%d\"...", ckpts[0])
                loaded = torch.load(directory / f"checkpoint_{ckpts[0]}.pt")
                net.load_state_dict(loaded["net"])
                trainer.load_state_dict(loaded["trainer"])
            else:
                self.logger.info(
                    "No usable checkpoint found. "
                    "Skipping checkpoint restoration."
                )


@logged
class LRScheduler(TrainingPlugin):

    r"""
    Reduce learning rate on loss plateau

    Parameters
    ----------
    *optims
        Optimizers
    monitor
        Loss to monitor
    patience
        Patience to reduce learning rate
    burnin
        Burn-in epochs to skip before initializing learning rate scheduling
    """

    def __init__(
            self, *optims: torch.optim.Optimizer, monitor: str = None,
            patience: int = None, burnin: int = 0, factor: float = 0.5
    ) -> None:
        super().__init__()
        if monitor is None:
            raise ValueError("`monitor` must be specified!")
        self.monitor = monitor
        if patience is None:
            raise ValueError("`patience` must be specified!")
        self.schedulers = [
            ReduceLROnPlateau(optim, patience=patience, factor=factor)
            for optim in optims
        ]
        # self.schedulers = [
        #     CosineAnnealingWarmRestarts(optim, T_0=patience, T_mult=2, verbose=True)
        #     for optim in optims
        # ]
        self.burnin = burnin

    def attach(
            self, net: torch.nn.Module, trainer: Trainer,
            train_engine: ignite.engine.Engine,
            val_engine: ignite.engine.Engine,
            train_loader: Iterable,
            val_loader: Optional[Iterable],
            directory: pathlib.Path
    ) -> None:
        score_engine = val_engine if val_engine else train_engine
        event_filter = lambda engine, event: event > self.burnin
        for scheduler in self.schedulers:
            scheduler.last_epoch = self.burnin
        train_engine.state.n_lrs = 0

        @train_engine.on(EPOCH_COMPLETED(event_filter=event_filter))  # pylint: disable=not-callable
        def _():
            update_flags = set()
            for scheduler in self.schedulers:
                old_lr = scheduler.optimizer.param_groups[0]["lr"]
                scheduler.step(score_engine.state.metrics[self.monitor])
                #scheduler.step()
                new_lr = scheduler.optimizer.param_groups[0]["lr"]
                update_flags.add(new_lr != old_lr)
            if len(update_flags) != 1:
                raise RuntimeError("Learning rates are out of sync!")
            if update_flags.pop():
                train_engine.state.n_lrs += 1
                self.logger.info("Learning rate reduction: step %d", train_engine.state.n_lrs)

def _embedding_only(
        adata: ad.AnnData, keys: Iterable[str] = ("X_glue", )
) -> ad.AnnData:
    r"""
    A lightweight copy of a dataset carrying only its annotations and selected
    embeddings, so that visualizations never touch (or load) ``X``

    Datasets can be backed by disk, in which case subsetting, copying or
    concatenating them would read the whole matrix into memory (or fail).

    Parameters
    ----------
    adata
        Dataset to summarize
    keys
        ``obsm`` keys to carry over (missing ones are ignored)

    Returns
    -------
    embedding
        Dataset with the same ``obs`` and the requested embeddings
    """
    return ad.AnnData(
        obs=adata.obs.copy(),
        obsm={key: adata.obsm[key] for key in keys if key in adata.obsm}
    )


@logged
class EmbeddingVisualizer(TrainingPlugin):

    r"""
    Metrics and figures of the embeddings every ``save_interval`` epochs

    Logs the same label transfer and clustering metrics to wandb as before and
    writes figures to ``out_dir`` (see :mod:`scglue.models.viz`):

    * ``cells/``, ``joint/``, ``confusion/``, ``params/``: one file per save
      step (numbered, for animations)
    * ``features/``, ``loci/``: overwritten at every save step; the feature
      embeddings of one chromosome (that of the first locus) and the loci of
      the first ``n_loci`` of ``genes`` only

    Parameters
    ----------
    genes
        Genes whose loci are drawn (the first ``n_loci`` found in the graph)
    level
        ``"full"`` (all figures), ``"basic"`` (cells, joint, confusion and
        parameters) or ``"none"`` (metrics only)
    max_umap_cells
        Cells per modality embedded for the joint figure
    max_locus_cells
        Cells per cell type pooled for the observed / predicted locus contacts
    """

    def __init__(
            self,
            rna, rna_data_config,
            hic, hic_data_config,
            graph, vertices,
            features_ad: ad.AnnData,
            atac=None, atac_data_config=None,
            methyl=None, methyl_data_config=None,
            latent_dim=64,
            prefix='pretrain',
            out_dir: str = 'tmp_imgs',
            save_interval: int = 10,
            genes: Optional[Iterable[str]] = None,
            n_loci: int = 3,
            level: str = "full",
            max_umap_cells: int = 30000,
            max_locus_cells: int = 200,
            max_decode_elements: float = 5e8  # unused, kept for compatibility
    ) -> None:
        super().__init__()
        self.rna = rna
        self.rna_data_config = rna_data_config
        self.hic = hic
        self.data_config = hic_data_config
        self.atac = atac
        self.atac_data_config = atac_data_config
        self.methyl = methyl
        self.methyl_data_config = methyl_data_config
        self.graph = graph
        self.vertices = pd.Index(vertices)
        self.features_ad = features_ad
        self.prefix = prefix
        self.out_dir = out_dir
        self.save_interval = save_interval
        self.genes = list(genes or [])
        self.n_loci = n_loci
        self.level = level
        self.max_umap_cells = max_umap_cells
        self.max_locus_cells = max_locus_cells
        self.seed = 0

        self.hic.obs['sorted_celltype'] = self.hic.obs['celltype'].astype(str)
        sorted_alpha = self.hic.obs_names.str.contains('alpha', case=False, regex=False, na=False)
        sorted_beta = self.hic.obs_names.str.contains('beta', case=False, regex=False, na=False)
        self.alpha_beta_val = bool(sorted_alpha.any() or sorted_beta.any())
        if self.alpha_beta_val:
            self.hic.obs.loc[sorted_alpha, 'sorted_celltype'] = 'alpha (sorted)'
            self.hic.obs.loc[sorted_beta, 'sorted_celltype'] = 'beta (sorted)'
        self.sorted_mask = np.asarray(sorted_alpha | sorted_beta)
        if level != "none":
            os.makedirs(self.out_dir, exist_ok=True)

    def attach(
            self, net: torch.nn.Module, trainer: Trainer,
            train_engine: ignite.engine.Engine,
            val_engine: ignite.engine.Engine,
            train_loader: Iterable,
            val_loader: Optional[Iterable],
            directory: pathlib.Path
    ) -> None:
        train_engine.state.n_lrs = 0

        @train_engine.on(EPOCH_COMPLETED)  # pylint: disable=not-callable
        def _():
            epoch = train_engine.state.epoch
            if epoch % self.save_interval == 0:
                was_training = net.training
                net.eval()
                try:
                    self.render(net, epoch, int(epoch // self.save_interval))
                finally:
                    net.train(was_training)

    #------------------------------- computation -------------------------------

    @torch.no_grad()
    def encode(self, net, key: str, adata: ad.AnnData, data_config) -> np.ndarray:
        # cap the fetch size so that wide (high resolution) data are not
        # densified in huge minibatches
        fetch = max(1, min(128, 2 ** 25 // max(adata.shape[1], 1)))
        data = AnnDataset([adata], [data_config], mode='eval', getitem_size=fetch)
        loader = DataLoader(
            data, batch_size=1, shuffle=False, num_workers=config.DATALOADER_NUM_WORKERS,
            pin_memory=config.DATALOADER_PIN_MEMORY and not config.CPU_ONLY, drop_last=False,
            persistent_workers=False
        )
        result = []
        for x, xrep, *_ in loader:
            u = net.x2u[key](x.to(net.device, non_blocking=True), xrep.to(net.device, non_blocking=True),
                             lazy_normalizer=True)[0]
            result.append(u.mean.detach().cpu())
        return torch.cat(result).numpy()

    @torch.no_grad()
    def feature_embeddings(self, net) -> torch.Tensor:
        graph = GraphDataset(self.graph, self.vertices)
        enorm = torch.as_tensor(normalize_edges(graph.eidx, graph.ewt), device=net.device)
        esgn = torch.as_tensor(graph.esgn, device=net.device)
        eidx = torch.as_tensor(graph.eidx, device=net.device)
        v = net.g2v(eidx, enorm, esgn).mean
        self.features_ad.X = v.detach().cpu().numpy()
        return v

    @staticmethod
    def batch_codes(adata: ad.AnnData, data_config) -> np.ndarray:
        use_batch = data_config.get("use_batch")
        batches = data_config.get("batches")
        if use_batch and batches is not None and len(batches) and use_batch in adata.obs:
            codes = pd.Index(batches).get_indexer(adata.obs[use_batch])
            return np.maximum(codes, 0)
        return np.zeros(adata.n_obs, dtype=int)

    @torch.no_grad()
    def decode(self, net, key: str, u: np.ndarray, v: torch.Tensor, b: np.ndarray, batch_size: int = 128):
        r"""
        Expected values (unit library size) in minibatches, as numpy arrays
        """
        for start in range(0, u.shape[0], batch_size):
            stop = min(start + batch_size, u.shape[0])
            u_ = torch.as_tensor(u[start:stop], dtype=torch.float32, device=net.device)
            b_ = torch.as_tensor(b[start:stop], dtype=torch.int64, device=net.device)
            l_ = torch.ones(stop - start, 1, device=net.device)
            yield start, stop, net.u2x[key](u_, v, b_, l_).mean.detach().cpu().numpy()

    def read_rows(self, rows: np.ndarray) -> np.ndarray:
        r"""
        Dense Hi-C counts of some cells (rows in increasing order), also for
        data backed on disk
        """
        layer = self.data_config.get("use_layer")
        if self.hic.isbacked or not layer:
            matrix = self.hic.X
        else:
            matrix = self.hic.layers[layer]
        sub = matrix[np.sort(rows)]
        if scipy.sparse.issparse(sub):
            sub = sub.toarray()
        return np.asarray(sub, dtype=np.float32)

    #--------------------------------- render ----------------------------------

    def render(self, net, epoch: int, save_i: int) -> None:
        self.hic.obsm['X_glue'] = self.encode(net, 'hic', self.hic, self.data_config)
        self.rna.obsm['X_glue'] = self.encode(net, 'rna', self.rna, self.rna_data_config)
        others = {}
        if self.atac is not None:
            self.atac.obsm['X_glue'] = self.encode(net, 'atac', self.atac, self.atac_data_config)
            others['atac'] = self.atac
        if self.methyl is not None:
            self.methyl.obsm['X_glue'] = self.encode(net, 'methyl', self.methyl, self.methyl_data_config)
            others['methyl'] = self.methyl
        v = self.feature_embeddings(net)

        log_dict = {"epoch": epoch}
        log_dict.update(self.label_metrics(self.hic, ""))
        for name, adata in others.items():
            log_dict.update(self.label_metrics(adata, f"{name}_"))
        sorted_result = self.sorted_validation() if self.alpha_beta_val else None
        if sorted_result is not None:
            log_dict.update(sorted_result[0])
        paired = self.paired_metrics(net, v)
        log_dict.update(paired[0])
        self.logger.info(
            f"ARI (Hi-C only): {log_dict.get('ari_' + self.prefix, 0):.2f}, "
            f"ARI (joint): {log_dict.get('ari_joint_' + self.prefix, 0):.2f}, "
            f"Accuracy (joint): {log_dict.get('accuracy_' + self.prefix, 0):.2f}, "
            f"Balanced Accuracy (joint): {log_dict.get('balanced_accuracy_' + self.prefix, 0):.2f}"
        )

        if self.level != "none":
            tag = f"{self.prefix}_{save_i}"
            steps = [
                ("cells", lambda: self.plot_cells(tag)),
                ("joint", lambda: self.plot_joint(tag, others, paired[1])),
                ("confusion", lambda: self.plot_confusion(tag, sorted_result)),
                ("params", lambda: log_dict.update(self.plot_params(net, v, tag))),
            ]
            if self.level == "full":
                steps += [
                    ("features", lambda: self.plot_features(net, v)),
                    ("loci", lambda: log_dict.update(self.plot_loci(net, v))),
                ]
            timings = []
            for name, step in steps:
                start = time.time()
                try:
                    step()
                except Exception as e:  # noqa: BLE001 - a figure must not stop training
                    self.logger.warning(f"Skipping {name} figure: {type(e).__name__}: {e}")
                timings.append(f"{name} {time.time() - start:.0f}s")
            self.logger.info("Figures: " + ", ".join(timings))
        try:
            wandb.log(log_dict, step=epoch)
        except Exception as e:  # noqa: BLE001
            self.logger.error(f"Error logging to wandb: {e}")

    #--------------------------------- metrics ---------------------------------

    def label_metrics(self, adata: ad.AnnData, prefix: str) -> dict:
        r"""
        Label transfer from RNA and clustering of one modality (labeled cells only)
        """
        p = self.prefix
        transfer_labels(self.rna, adata, "celltype", use_rep="X_glue", n_neighbors=10,
                        key_added="pred_celltype", metric="cosine")
        sc.pp.neighbors(adata, use_rep="X_glue", metric="cosine")
        sc.tl.leiden(adata)
        x = adata.obsm["X_glue"]
        truth = adata.obs["celltype"].astype(str).to_numpy()
        pred = adata.obs["pred_celltype"].astype(str).to_numpy()
        leiden = adata.obs["leiden"].astype(str).to_numpy()
        labeled = ~np.isin(truth, viz.UNLABELED)
        if labeled.sum() < 2:
            return {}
        n_types = len(np.unique(truth[labeled]))
        if labeled.sum() <= 15000:
            agglo = AgglomerativeClustering(n_clusters=n_types).fit_predict(x[labeled])
        else:  # quadratic memory
            agglo = leiden[labeled]
        adata.obs["agglomerative"] = "NA"
        adata.obs.loc[labeled, "agglomerative"] = np.asarray(agglo).astype(str)
        # predictions outside the query vocabulary count as wrong
        vocab = set(truth[labeled])
        pred_q = np.where(np.isin(pred, list(vocab)), pred, "Other")

        def sil(labels):
            try:
                return silhouette_score(x[labeled], labels, sample_size=min(5000, int(labeled.sum())),
                                        random_state=self.seed)
            except ValueError:
                return 0.0

        t, q = truth[labeled], pred_q[labeled]
        labels = sorted(vocab | {"Other"})
        return {
            f"{prefix}ari_{p}": adjusted_rand_score(t, agglo),
            f"{prefix}ari_leiden_{p}": adjusted_rand_score(t, leiden[labeled]),
            f"{prefix}ari_joint_{p}": adjusted_rand_score(t, q),
            f"{prefix}f1_micro_{p}": f1_score(t, q, average="micro", labels=labels),
            f"{prefix}f1_macro_{p}": f1_score(t, q, average="macro", labels=sorted(vocab), zero_division=0),
            f"{prefix}accuracy_{p}": accuracy_score(t, q),
            f"{prefix}balanced_accuracy_{p}": balanced_accuracy_score(t, q),
            f"{prefix}silhouette_leiden_{p}": sil(leiden[labeled]),
            f"{prefix}silhouette_kmeans_{p}": sil(agglo),
            f"{prefix}silhouette_joint_{p}": sil(q),
            f"{prefix}silhouette_celltype_{p}": sil(t),
        }

    def sorted_validation(self):
        r"""
        FACS-sorted alpha / beta cells, labeled from the alpha / beta RNA cells only
        """
        p = self.prefix
        hic = _embedding_only(self.hic)[self.sorted_mask].copy()
        rna = _embedding_only(self.rna)[self.rna.obs['celltype'].astype(str).isin(['Alpha', 'Beta'])].copy()
        if hic.n_obs < 2 or rna.n_obs < 2:
            return None
        transfer_labels(rna, hic, "celltype", use_rep="X_glue", n_neighbors=5,
                        key_added="pred_celltype_sorted", metric="cosine")
        t = hic.obs["celltype"].astype(str).to_numpy()
        q = hic.obs["pred_celltype_sorted"].astype(str).to_numpy()
        try:
            asw = silhouette_score(hic.obsm["X_glue"], t)
        except ValueError:
            asw = 0.0
        metrics = {
            f"val_accuracy_{p}": accuracy_score(t, q),
            f"val_balanced_accuracy_{p}": balanced_accuracy_score(t, q),
            f"val_ari_{p}": adjusted_rand_score(t, q),
            f"val_celltype_asw_{p}": asw,
        }
        self.logger.info(f"Val Accuracy (joint): {metrics[f'val_accuracy_{p}']:.2f}, "
                         f"Val ARI (joint): {metrics[f'val_ari_{p}']:.2f}")
        return metrics, (t, q)

    def paired_metrics(self, net, v: torch.Tensor):
        r"""
        Distance and correlation of the embeddings of cells profiled in both
        modalities, and of the RNA predicted from either embedding
        """
        p = self.prefix
        out = {f"avg_paired_dist_{p}": 0, f"avg_corr_{p}": 0, f"avg_expression_corr_{p}": 0}
        shared = pd.Index(self.hic.obs_names).intersection(self.rna.obs_names)
        if shared.empty:
            return out, None
        if shared.size > 2000:
            shared = pd.Index(np.random.RandomState(self.seed).choice(shared, 2000, replace=False))
        hic_z = self.hic.obsm['X_glue'][pd.Index(self.hic.obs_names).get_indexer(shared)]
        rna_z = self.rna.obsm['X_glue'][pd.Index(self.rna.obs_names).get_indexer(shared)]
        out[f"avg_paired_dist_{p}"] = float(np.linalg.norm(hic_z - rna_z, axis=1).mean())
        out[f"avg_corr_{p}"] = float(np.mean([pearsonr(a, b)[0] for a, b in zip(hic_z, rna_z)]))
        v_rna = v[getattr(net, "rna_idx")]
        b = np.zeros(len(shared), dtype=int)
        from_hic = np.vstack([m for *_, m in self.decode(net, 'rna', hic_z, v_rna, b)])
        from_rna = np.vstack([m for *_, m in self.decode(net, 'rna', rna_z, v_rna, b)])
        out[f"avg_expression_corr_{p}"] = float(np.mean([pearsonr(a, b_)[0] for a, b_ in zip(from_hic, from_rna)]))

        def cos_dist(a, b_):
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            b_ = b_ / np.linalg.norm(b_, axis=1, keepdims=True)
            return 1 - (a * b_).sum(1)

        shuffled = rna_z[np.random.RandomState(self.seed).permutation(len(rna_z))]
        return out, (cos_dist(hic_z, rna_z), cos_dist(hic_z, shuffled))

    #--------------------------------- figures ---------------------------------

    def path(self, *parts: str) -> str:
        return os.path.join(self.out_dir, *parts)

    def label_palette(self) -> dict:
        groups = [self.hic.obs["sorted_celltype"], self.hic.obs["celltype"], self.rna.obs["celltype"],
                  self.hic.obs["pred_celltype"]]
        return viz.palette_for(*groups)

    def plot_cells(self, tag: str) -> None:
        hic = _embedding_only(self.hic)
        hic.obsm["X_pca"] = PCA(n_components=2).fit_transform(hic.obsm["X_glue"])
        sc.pp.neighbors(hic, use_rep="X_glue", metric="cosine")
        sc.tl.umap(hic, random_state=self.seed)
        obs = hic.obs.copy()
        obs["batch"] = obs["batch"].astype(str) if "batch" in obs else "NA"
        truth_key = "sorted_celltype" if self.alpha_beta_val else "celltype"
        columns = [(truth_key, "Annotation", True), ("pred_celltype", "Transferred from RNA", True),
                   ("leiden", "Leiden", False), ("depth", "Depth (fraction of max)", False),
                   ("balancing_weight" if "balancing_weight" in obs else "batch", "Batch", False)]
        viz.plot_cells(obs, {"UMAP": hic.obsm["X_umap"], "PCA": hic.obsm["X_pca"]}, columns,
                       self.label_palette(), self.path("cells", f"{tag}.png"),
                       f"Hi-C cells ({self.prefix}, {hic.n_obs:,} cells)")

    def plot_joint(self, tag: str, others: dict, paired) -> None:
        rs = np.random.RandomState(self.seed)
        parts = []
        for name, adata in [("hic", self.hic), ("rna", self.rna), *others.items()]:
            light = _embedding_only(adata)
            if light.n_obs > self.max_umap_cells:
                light = light[np.sort(rs.choice(light.n_obs, self.max_umap_cells, replace=False))].copy()
            light.obs = pd.DataFrame({
                "domain": name,
                "label": light.obs["celltype"].astype(str).to_numpy(),
                "pred": light.obs["pred_celltype" if name != "rna" else "celltype"].astype(str).to_numpy(),
            }, index=light.obs_names)
            parts.append(light)
        combined = ad.concat(parts)
        combined.obs_names_make_unique()  # paired cells share names across modalities
        combined.obsm["X_pca"] = PCA(n_components=2).fit_transform(combined.obsm["X_glue"])
        sc.pp.neighbors(combined, use_rep="X_glue", metric="cosine")
        sc.tl.umap(combined, random_state=self.seed)
        columns = [("domain", "Modality", False), ("label", "Annotation", True), ("pred", "Prediction (RNA: own)", True)]
        extra = (lambda ax: viz.plot_paired(ax, *paired)) if paired is not None else None
        viz.plot_cells(combined.obs, {"UMAP": combined.obsm["X_umap"], "PCA": combined.obsm["X_pca"]}, columns,
                       self.label_palette(), self.path("joint", f"{tag}.png"),
                       f"Joint embedding ({self.prefix}; up to {self.max_umap_cells:,} cells per modality)",
                       extra=extra)

    def plot_confusion(self, tag: str, sorted_result) -> None:
        truth = self.hic.obs["celltype"].astype(str).to_numpy()
        pred = self.hic.obs["pred_celltype"].astype(str).to_numpy()
        labeled = ~np.isin(truth, viz.UNLABELED)
        panels = [(truth[labeled], pred[labeled], "Hi-C, labeled cells")]
        if sorted_result is not None:
            panels.append((*sorted_result[1], "Sorted cells (alpha/beta reference)"))
        if self.atac is not None:
            panels.append((self.atac.obs["celltype"].astype(str).to_numpy(),
                           self.atac.obs["pred_celltype"].astype(str).to_numpy(), "ATAC"))
        viz.plot_confusion(panels, self.path("confusion", f"{tag}.png"))

    def plot_params(self, net, v: torch.Tensor, tag: str) -> dict:
        view = viz.DecoderView(net.u2x['hic'])
        res_i = view.finest()
        name, offset, n_anchors, _ = view.specs[res_i]
        v_res = v[getattr(net, "hic_idx")][offset:offset + n_anchors]
        u = torch.as_tensor(self.hic.obsm["X_glue"], dtype=torch.float32, device=net.device)
        labels = self.hic.obs["celltype"].astype(str).to_numpy()
        if np.isin(labels, viz.UNLABELED).mean() > 0.5:
            labels = self.hic.obs["pred_celltype"].astype(str).to_numpy()
        metrics, attn = {}, None
        if view.use_attn:
            attn = viz.attention_summary(view, v_res, res_i, seed=self.seed)
            if n_anchors <= 50000:  # the attention maps must reproduce the module output
                module, qk, val = view.attention_inputs(res_i, v_res, 1)
                err = viz.check_attention(module, qk, qk, val)
                metrics[f"attention_check_error_{self.prefix}"] = err
                if err > 1e-3:
                    self.logger.warning(f"Attention maps deviate from the decoder output by {err:.2e}")
        keys = view.stratum_keys(res_i, v_res) if not view.use_conv else None
        viz.plot_params(view, u, labels, attn, keys, self.path("params", f"{tag}.png"),
                        f"Decoder parameters ({self.prefix})")
        return metrics

    def anchor_table(self, net, view, res_i: int) -> pd.DataFrame:
        name, offset, n_anchors, _ = view.specs[res_i]
        idx = getattr(net, "hic_idx")
        idx = idx.detach().cpu().numpy() if torch.is_tensor(idx) else np.asarray(idx)
        names = self.vertices[idx[offset:offset + n_anchors]].astype(str)
        parsed = names.str.extract(r"^(?P<chrom>.+):(?P<start>\d+)-(?P<end>\d+)$")
        parsed["start"] = pd.to_numeric(parsed["start"]).astype("Int64")
        parsed["end"] = pd.to_numeric(parsed["end"]).astype("Int64")
        parsed.index = names
        return parsed

    def locus_genes(self, anchors: pd.DataFrame) -> List[Tuple[str, str, int]]:
        r"""
        (gene, chromosome, TSS) of the first ``n_loci`` requested genes found
        in the graph and on an anchored chromosome
        """
        var = self.rna.var
        lookup = {str(g).lower(): g for g in var.index}
        chroms = set(anchors["chrom"].dropna())
        found = []
        candidates = self.genes or list(var.index[var.get("highly_variable", pd.Series(True, index=var.index))][:200])
        for g in candidates:
            g = lookup.get(str(g).lower())
            if g is None or g not in self.vertices or "chrom" not in var:
                continue
            row = var.loc[g]
            chrom = str(row["chrom"])
            if chrom not in chroms:
                alt = chrom[3:] if chrom.startswith("chr") else f"chr{chrom}"
                if alt not in chroms:
                    continue
                chrom = alt
            tss = int(row["chromEnd"] if str(row.get("strand", "+")) == "-" else row["chromStart"])
            found.append((g, chrom, tss))
            if len(found) >= self.n_loci:
                break
        return found

    def plot_features(self, net, v: torch.Tensor) -> None:
        view = viz.DecoderView(net.u2x['hic'])
        res_i = view.finest()
        anchors = self.anchor_table(net, view, res_i)
        loci = self.locus_genes(anchors)
        chrom = loci[0][1] if loci else anchors["chrom"].value_counts().index[0]
        on_chrom = np.where((anchors["chrom"] == chrom).to_numpy())[0]
        name, offset, n_anchors, _ = view.specs[res_i]
        v_res = v[getattr(net, "hic_idx")][offset:offset + n_anchors].detach().cpu().numpy()
        # the other features on this chromosome (genes, peaks, ...), from the graph
        obs = self.features_ad.obs
        chrom_alt = chrom[3:] if chrom.startswith("chr") else f"chr{chrom}"
        other = ((obs["type"].astype(str) != "Hi-C") & obs["chrom"].astype(str).isin([chrom, chrom_alt])).to_numpy()
        viz.plot_feature_chrom(v_res[on_chrom], anchors["start"].to_numpy(dtype=float)[on_chrom], chrom,
                               self.features_ad.X[other], obs.loc[other, "type"].astype(str).to_numpy(),
                               obs.loc[other, "pos"].astype(float).to_numpy(),
                               self.path("features", f"{self.prefix}_chrom_{chrom}.png"),
                               f"Feature embeddings on {chrom} ({self.prefix}, {name})", seed=self.seed)

    def plot_loci(self, net, v: torch.Tensor) -> dict:
        view = viz.DecoderView(net.u2x['hic'])
        res_i = view.finest()
        name, offset, n_anchors, n_strata = view.specs[res_i]
        anchors = self.anchor_table(net, view, res_i)
        loci = self.locus_genes(anchors)
        if not loci:
            return {}
        v_res = v[getattr(net, "hic_idx")][offset:offset + n_anchors]
        band = view.bands[res_i]
        res_cols = band[band >= 0]
        flank = n_strata + 5

        # cell pools: annotated types if most cells are labeled, else predictions
        labels = self.hic.obs["celltype"].astype(str).to_numpy()
        if np.isin(labels, viz.UNLABELED).mean() > 0.5:
            labels = self.hic.obs["pred_celltype"].astype(str).to_numpy()
        counts = pd.Series(labels[~np.isin(labels, viz.UNLABELED)]).value_counts()
        types = list(counts.index[:3])
        rs = np.random.RandomState(self.seed)
        pools = {t: np.sort(rs.choice(np.where(labels == t)[0], min(self.max_locus_cells, int(counts[t])), replace=False))
                 for t in types}

        # observed and predicted contacts of the pooled cells, normalized per
        # cell, accumulated per type; expected values per stratum over all anchors
        v_hic = v[getattr(net, "hic_idx")]
        n_features = self.hic.shape[1]
        batch_size = max(8, min(128, int(2 ** 25 // max(n_features, 1))))
        b_all = self.batch_codes(self.hic, self.data_config)
        sums = {t: {"obs": np.zeros(n_features), "pred": np.zeros(n_features)} for t in types}
        strat_obs, strat_pred, n_total = np.zeros(n_strata), np.zeros(n_strata), 0
        strat_cols = [band[k][band[k] >= 0] for k in range(n_strata)]
        for t in types:
            rows = pools[t]
            for start in range(0, len(rows), batch_size):
                chunk = rows[start:start + batch_size]
                x = self.read_rows(chunk)
                lib = x[:, res_cols].sum(1, keepdims=True)
                x = x / np.maximum(lib, 1)
                sums[t]["obs"] += x.sum(0)
                _, _, mu = next(self.decode(net, 'hic', self.hic.obsm["X_glue"][chunk], v_hic, b_all[chunk],
                                            batch_size=len(chunk)))
                mu = mu / np.maximum(mu[:, res_cols].sum(1, keepdims=True), 1e-12)
                sums[t]["pred"] += mu.sum(0)
                strat_obs += np.array([x[:, c].mean(1).sum() for c in strat_cols])
                strat_pred += np.array([mu[:, c].mean(1).sum() for c in strat_cols])
                n_total += len(chunk)
        exp_obs, exp_pred = strat_obs / max(n_total, 1), strat_pred / max(n_total, 1)

        metrics, attn_corrs = {}, []
        for gene, chrom, tss in loci:
            on_chrom = np.where((anchors["chrom"] == chrom).to_numpy())[0]
            starts = anchors["start"].to_numpy(dtype=float)[on_chrom]
            ends = anchors["end"].to_numpy(dtype=float)[on_chrom]
            inside = np.where((starts <= tss) & (ends > tss))[0]
            c = int(inside[0]) if inside.size else int(np.argmin(np.abs(starts - tss)))
            lo, hi = max(0, c - flank), min(len(on_chrom), c + flank + 1)
            w_idx = on_chrom[lo:hi]  # sequence positions of the window anchors
            center = c - lo
            n = len(w_idx)
            w_names = list(anchors.index[w_idx])
            pos_of = {a: i for i, a in enumerate(w_names)}

            edges, linked = {}, {}
            for a in w_names:
                if a not in self.graph:
                    continue
                for _, other, data in self.graph.edges(a, data=True):
                    if other in pos_of and other != a and data.get("type", "hic") in ("hic", "trans"):
                        i, j = sorted((pos_of[a], pos_of[other]))
                        if j - i > 1:  # adjacency edges share the "hic" type and are not drawn
                            edges[(i, j)] = max(edges.get((i, j), 0.0), float(data.get("weight", 1.0)))
                    elif self.graph.nodes[other].get("feature_type") == "RNA" and other in self.rna.var.index:
                        gpos = self.rna.var.loc[other, "chromStart"] if "chromStart" in self.rna.var else np.nan
                        linked[other] = (float(gpos), pos_of[a])
            edges = pd.DataFrame([(i, j, w) for (i, j), w in edges.items()], columns=["i", "j", "weight"])
            linked = pd.DataFrame([(g_, p_, a_) for g_, (p_, a_) in linked.items()], columns=["name", "pos", "anchor"])
            if len(linked) > 15:  # the gene itself, then the nearest ones
                linked["d"] = (linked["pos"] - tss).abs() + np.where(linked["name"] == gene, -1e12, 0)
                linked = linked.nsmallest(15, "d").drop(columns="d")

            pool_bands = []
            window_cols = band[:, w_idx]
            for t in types:
                def to_band(values, expected):
                    out = np.full(window_cols.shape, np.nan)
                    ok = window_cols >= 0
                    out[ok] = values[window_cols[ok]] / len(pools[t])
                    return out / np.maximum(expected[:, None], 1e-12)
                pool_bands.append((t, len(pools[t]), to_band(sums[t]["obs"], exp_obs),
                                   to_band(sums[t]["pred"], exp_pred)))

            attn = None
            if view.use_attn:
                mean_attn = np.zeros((n, n))
                gene_rows = np.zeros((n_strata - 1, n))
                for k in range(1, n_strata):
                    module, qk, _ = view.attention_inputs(res_i, v_res, k)
                    W = module.window_size
                    windows = sorted(set((w_idx // W).tolist()))
                    weights, key_pos, _ = viz.local_attention_weights(module, qk, qk, windows)
                    dense = viz.dense_attention(weights, key_pos, W, windows, w_idx)
                    mean_attn += dense / (n_strata - 1)
                    gene_rows[k - 1] = dense[center]
                attn = (mean_attn, gene_rows)
            vw = (v_res[w_idx] - v_res.mean(0)).detach().cpu().numpy()  # centered on the genome-wide mean
            vw = vw / np.maximum(np.linalg.norm(vw, axis=1, keepdims=True), 1e-8)
            m = viz.plot_locus(gene, chrom, anchors["start"].to_numpy(dtype=float)[w_idx],
                               anchors["end"].to_numpy(dtype=float)[w_idx], tss, center, edges, linked,
                               pool_bands, attn, vw @ vw.T, self.path("loci", f"{self.prefix}_{gene}.png"),
                               f"{gene} locus ({self.prefix}, {name}, ±{flank} anchors)")
            if "attention_contact_spearman" in m:
                attn_corrs.append(m["attention_contact_spearman"])
        if attn_corrs:
            metrics[f"attention_contact_spearman_{self.prefix}"] = float(np.nanmean(attn_corrs))
        return metrics
