r"""
Multi-resolution, memory-bounded Hi-C feature construction for HiGLUE.

The single-resolution pipeline in :mod:`preprocess_data` materializes a dense
``cells x (n_strata * n_bins)`` matrix, which is only feasible at coarse
resolution. This module builds the same kind of band representation, but

* at several resolutions at once (each cell ends up with one row spanning all
  resolutions, which the model encodes into a multi-resolution embedding),
* by streaming cells one at a time so that peak memory is independent of the
  number of cells,
* while filtering the (potentially enormous) high resolution feature space down
  to a budget of informative anchors before any per-cell data is written.

Coarser grids are derived from the finest ``.scool`` on the fly by re-binning
genomic coordinates, so only one ``.scool`` file is required.
"""

import os
import re
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
import scipy.sparse
from tqdm.auto import tqdm


RES_PATTERN = re.compile(r"^([0-9]*\.?[0-9]+)\s*(bp|b|kb|mb)?$", re.IGNORECASE)
RES_UNITS = {"bp": 1, "b": 1, "kb": 1_000, "mb": 1_000_000}


def parse_resolution(name) -> int:
    r"""
    Convert a resolution name into a bin size in base pairs

    Parameters
    ----------
    name
        Resolution name, e.g. ``"500kb"``, ``"5kb"``, ``"1mb"`` or ``50000``

    Returns
    -------
    binsize
        Bin size in base pairs
    """
    if isinstance(name, (int, np.integer)):
        return int(name)
    match = RES_PATTERN.match(str(name).strip())
    if match is None:
        raise ValueError(f"Cannot interpret resolution '{name}'!")
    value, unit = match.groups()
    return int(round(float(value) * RES_UNITS[(unit or "bp").lower()]))


def format_resolution(binsize: int) -> str:
    r"""
    Convert a bin size in base pairs into a resolution name
    """
    binsize = int(binsize)
    if binsize % 1_000_000 == 0:
        return f"{binsize // 1_000_000}mb"
    if binsize % 1_000 == 0:
        return f"{binsize // 1_000}kb"
    return f"{binsize}bp"


def broadcast_option(values, n: int, name: str, default=None) -> List:
    r"""
    Expand a per-resolution command line option to one value per resolution

    Parameters
    ----------
    values
        Either ``None``, a single value, or one value per resolution
    n
        Number of resolutions
    name
        Option name (used in error messages)
    default
        Value used when ``values`` is ``None``

    Returns
    -------
    values
        One value per resolution
    """
    if values is None:
        return [default] * n
    if not isinstance(values, (list, tuple)):
        values = [values]
    values = list(values)
    if len(values) == 1:
        return values * n
    if len(values) != n:
        raise ValueError(
            f"`{name}` must have either 1 or {n} values, got {len(values)}!"
        )
    return values


#--------------------------------- Bin grids -----------------------------------

class ResolutionGrid:

    r"""
    Genomic bin grid at a single resolution, together with the mapping from the
    bins of the source ``.scool`` file.

    Parameters
    ----------
    name
        Resolution name
    binsize
        Bin size in base pairs (``0`` for a variable-width source grid)
    bins
        Bin table with ``chrom``, ``chromStart``, ``chromEnd`` and ``name``
    base_to_bin
        Mapping from source bin index to grid bin index (``-1`` when the source
        bin is dropped, e.g. on an excluded chromosome)
    """

    def __init__(
            self, name: str, binsize: int, bins: pd.DataFrame,
            base_to_bin: np.ndarray
    ) -> None:
        self.name = str(name)
        self.binsize = int(binsize)
        self.bins = bins.reset_index(drop=True)
        self.base_to_bin = np.asarray(base_to_bin, dtype=np.int64)
        self.chrom_code = pd.Categorical(
            self.bins["chrom"], categories=pd.unique(self.bins["chrom"])
        ).codes.astype(np.int32)

    @property
    def n_bins(self) -> int:
        r"""
        Number of bins
        """
        return self.bins.shape[0]

    @property
    def names(self) -> np.ndarray:
        r"""
        Bin names (``chrom:start-end``)
        """
        return self.bins["name"].to_numpy()

    @classmethod
    def from_base(
            cls, name: str, base_bins: pd.DataFrame, keep_mask: np.ndarray
    ) -> "ResolutionGrid":
        r"""
        Use the bins of the source ``.scool`` file as-is
        """
        bins = base_bins.loc[keep_mask].reset_index(drop=True)
        base_to_bin = np.full(base_bins.shape[0], -1, dtype=np.int64)
        base_to_bin[np.where(keep_mask)[0]] = np.arange(bins.shape[0])
        return cls(name, 0, bins, base_to_bin)

    @classmethod
    def uniform(
            cls, name: str, binsize: int, chromsizes: Mapping[str, int],
            base_bins: pd.DataFrame, keep_mask: np.ndarray
    ) -> "ResolutionGrid":
        r"""
        Build a uniform grid and map source bins onto it by start coordinate
        """
        chroms, starts, ends, offsets = [], [], [], {}
        offset = 0
        for chrom in chromsizes:
            size = int(chromsizes[chrom])
            n = max(1, int(np.ceil(size / binsize)))
            start = np.arange(n, dtype=np.int64) * binsize
            # The last bin deliberately runs past the end of the chromosome.
            # Clipping it to the chromosome length would let two resolutions
            # produce the same bin name (e.g. the last 500kb and 200kb bins of
            # a chromosome both clip to `chr2:182000000-182113224`), and bin
            # names are what identify features and graph vertices.
            end = start + binsize
            chroms.append(np.full(n, chrom, dtype=object))
            starts.append(start)
            ends.append(end)
            offsets[chrom] = offset
            offset += n
        bins = pd.DataFrame({
            "chrom": np.concatenate(chroms),
            "chromStart": np.concatenate(starts),
            "chromEnd": np.concatenate(ends)
        })
        bins["name"] = [
            f"{c}:{s}-{e}" for c, s, e
            in zip(bins["chrom"], bins["chromStart"], bins["chromEnd"])
        ]
        base_offset = base_bins["chrom"].map(offsets).to_numpy()
        base_to_bin = np.where(
            keep_mask & ~pd.isna(base_offset),
            np.nan_to_num(base_offset, nan=0).astype(np.int64)
            + base_bins["chromStart"].to_numpy() // binsize,
            -1
        ).astype(np.int64)
        return cls(name, binsize, bins, base_to_bin)


def build_grids(
        base_bins: pd.DataFrame, chromsizes: Mapping[str, int],
        resolutions: Sequence[str], base_resolution: Optional[str] = None,
        use_xy: bool = False
) -> "Dict[str, ResolutionGrid]":
    r"""
    Build one bin grid per requested resolution

    Parameters
    ----------
    base_bins
        Bin table of the source ``.scool`` file
        (``chrom``, ``chromStart``, ``chromEnd``)
    chromsizes
        Chromosome sizes, in the order they should appear in the grids
    resolutions
        Requested resolution names, coarse to fine
    base_resolution
        Nominal resolution of the source ``.scool`` file. The grid of a
        resolution matching it is taken directly from the source bins, which
        preserves variable-width binning.
    use_xy
        Whether to keep sex chromosomes

    Returns
    -------
    grids
        Bin grid per resolution
    """
    keep_mask = np.ones(base_bins.shape[0], dtype=bool)
    if not use_xy:
        sex = base_bins["chrom"].astype(str).str.lower().str.contains("x|y")
        keep_mask &= ~sex.to_numpy()
        chromsizes = {
            chrom: size for chrom, size in chromsizes.items()
            if "x" not in chrom.lower() and "y" not in chrom.lower()
        }
    keep_mask &= base_bins["chrom"].isin(list(chromsizes)).to_numpy()
    base_binsize = parse_resolution(base_resolution) if base_resolution else None

    grids = {}
    for res in resolutions:
        binsize = parse_resolution(res)
        if base_binsize is not None and binsize < base_binsize:
            raise ValueError(
                f"Requested resolution '{res}' is finer than the source data "
                f"('{base_resolution}'). Pass the finest .scool file to SCORE."
            )
        if base_binsize is not None and binsize == base_binsize:
            grids[res] = ResolutionGrid.from_base(res, base_bins, keep_mask)
        else:
            grids[res] = ResolutionGrid.uniform(
                res, binsize, chromsizes, base_bins, keep_mask
            )
    return grids


#------------------------------ Streaming access -------------------------------

class ScoolReader:

    r"""
    Direct reader for the pixel tables of a ``.scool`` file

    Reading the HDF5 datasets directly (instead of instantiating a
    :class:`cooler.Cooler` per cell) keeps a full pass over tens of thousands
    of high resolution cells affordable.

    Parameters
    ----------
    scool_file
        Path to the ``.scool`` file
    res_name
        Resolution suffix used in cell names
    """

    def __init__(self, scool_file: os.PathLike, res_name: str = "") -> None:
        self.scool_file = str(scool_file)
        self.res_name = res_name
        self.handle = h5py.File(self.scool_file, "r")
        self.available = set(self.handle["cells"].keys())

    def close(self) -> None:
        r"""
        Close the underlying file
        """
        self.handle.close()

    def __enter__(self) -> "ScoolReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def resolve(self, cell: str) -> Optional[str]:
        r"""
        Resolve a cell name against the names present in the file
        """
        candidates = [cell]
        if self.res_name:
            candidates += [
                cell.replace(self.res_name, f"comb.{self.res_name}"),
                cell.replace(self.res_name, f"3C.{self.res_name}"),
                f"{cell}.{self.res_name}"
            ]
        for candidate in candidates:
            if candidate in self.available:
                return candidate
        return None

    def bins(self) -> pd.DataFrame:
        r"""
        Bin table shared by all cells
        """
        cell = sorted(self.available)[0]
        group = self.handle[f"cells/{cell}"]
        chrom_names = np.asarray(group["chroms/name"][:])
        chrom_names = np.array([
            name.decode() if isinstance(name, bytes) else str(name)
            for name in chrom_names
        ])
        bins = pd.DataFrame({
            "chrom": chrom_names[group["bins/chrom"][:]],
            "chromStart": group["bins/start"][:].astype(np.int64),
            "chromEnd": group["bins/end"][:].astype(np.int64)
        })
        bins["name"] = [
            f"{c}:{s}-{e}" for c, s, e
            in zip(bins["chrom"], bins["chromStart"], bins["chromEnd"])
        ]
        return bins

    def chromsizes(self) -> "Dict[str, int]":
        r"""
        Chromosome sizes, in file order
        """
        cell = sorted(self.available)[0]
        group = self.handle[f"cells/{cell}"]
        names = [
            name.decode() if isinstance(name, bytes) else str(name)
            for name in group["chroms/name"][:]
        ]
        lengths = group["chroms/length"][:].astype(np.int64)
        return dict(zip(names, lengths))

    def pixels(self, cell: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""
        Read the pixel table of one cell

        Returns
        -------
        bin1
            First bin index of every pixel
        bin2
            Second bin index of every pixel
        count
            Contact count of every pixel
        """
        name = self.resolve(cell)
        if name is None:
            raise KeyError(f"Cell '{cell}' not found in {self.scool_file}!")
        group = self.handle[f"cells/{name}/pixels"]
        return (
            group["bin1_id"][:].astype(np.int64),
            group["bin2_id"][:].astype(np.int64),
            group["count"][:].astype(np.float32)
        )


def band_columns(
        grid: ResolutionGrid, n_strata: int,
        bin1: np.ndarray, bin2: np.ndarray, count: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Project pixels onto the band representation of a grid

    Parameters
    ----------
    grid
        Bin grid
    n_strata
        Number of diagonal strata to keep
    bin1, bin2, count
        Pixel table of one cell, in source bin indices

    Returns
    -------
    columns
        Unique band positions (``stratum * n_bins + anchor``)
    values
        Aggregated contact counts of those positions
    """
    a1 = grid.base_to_bin[bin1]
    a2 = grid.base_to_bin[bin2]
    valid = (a1 >= 0) & (a2 >= 0)
    a1, a2, count = a1[valid], a2[valid], count[valid]
    valid = grid.chrom_code[a1] == grid.chrom_code[a2]  # cis only
    a1, a2, count = a1[valid], a2[valid], count[valid]
    lo = np.minimum(a1, a2)
    stratum = np.abs(a2 - a1)
    valid = stratum < n_strata
    lo, stratum, count = lo[valid], stratum[valid], count[valid]
    if not lo.size:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    flat = stratum * grid.n_bins + lo
    columns, inverse = np.unique(flat, return_inverse=True)
    values = np.bincount(inverse, weights=count).astype(np.float32)
    return columns, values


class BandStats:

    r"""
    Streaming statistics of a band representation

    Accumulates per-feature bulk counts (used to rank contacts) and per-anchor
    mean / variance / detection rate (used to rank anchors), all in space
    proportional to the band size rather than to the number of cells.

    Parameters
    ----------
    grid
        Bin grid
    n_strata
        Number of diagonal strata
    """

    def __init__(self, grid: ResolutionGrid, n_strata: int) -> None:
        self.grid = grid
        self.n_strata = int(n_strata)
        self.n_cells = 0
        self.feature_sum = np.zeros(self.n_strata * grid.n_bins, dtype=np.float32)
        self.anchor_sum = np.zeros(grid.n_bins, dtype=np.float64)
        self.anchor_sumsq = np.zeros(grid.n_bins, dtype=np.float64)
        self.anchor_cells = np.zeros(grid.n_bins, dtype=np.int32)
        self.depths = []

    def update(self, columns: np.ndarray, values: np.ndarray) -> None:
        r"""
        Accumulate the band representation of one cell
        """
        self.n_cells += 1
        self.depths.append(float(values.sum()))
        if not columns.size:
            return
        self.feature_sum[columns] += values  # `columns` is unique
        anchor = np.bincount(
            columns % self.grid.n_bins, weights=values,
            minlength=self.grid.n_bins
        )
        self.anchor_sum += anchor
        self.anchor_sumsq += anchor ** 2
        self.anchor_cells += anchor > 0

    def anchor_scores(self) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Per-anchor index of dispersion and detection count

        Returns
        -------
        dispersion
            Variance / mean of the per-anchor band coverage
        cells
            Number of cells in which the anchor was detected
        """
        n = max(self.n_cells, 1)
        mean = self.anchor_sum / n
        var = np.maximum(self.anchor_sumsq / n - mean ** 2, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            dispersion = np.where(mean > 0, var / mean, 0.0)
        return dispersion, self.anchor_cells


class TransStats:

    r"""
    Streaming pseudobulk statistics of trans-chromosomal contacts

    Kept at a single (coarse) resolution: a dense trans matrix is quadratic in
    the number of bins, so it is only affordable where bins are large.

    Parameters
    ----------
    grid
        Bin grid
    max_elements
        Refuse to allocate a matrix larger than this
    """

    def __init__(self, grid: ResolutionGrid, max_elements: float = 5e7) -> None:
        self.grid = grid
        self.n_cells = 0
        if grid.n_bins ** 2 > max_elements:
            raise ValueError(
                f"A trans matrix at resolution '{grid.name}' would need "
                f"{grid.n_bins ** 2:,} entries; use a coarser resolution for "
                f"trans contacts or raise `max_elements`."
            )
        self.matrix = np.zeros((grid.n_bins, grid.n_bins), dtype=np.float32)

    def update(
            self, bin1: np.ndarray, bin2: np.ndarray, count: np.ndarray
    ) -> None:
        r"""
        Accumulate the trans contacts of one cell
        """
        self.n_cells += 1
        a1 = self.grid.base_to_bin[bin1]
        a2 = self.grid.base_to_bin[bin2]
        valid = (a1 >= 0) & (a2 >= 0)
        a1, a2, count = a1[valid], a2[valid], count[valid]
        trans = self.grid.chrom_code[a1] != self.grid.chrom_code[a2]
        a1, a2, count = a1[trans], a2[trans], count[trans]
        if not a1.size:
            return
        lo = np.minimum(a1, a2)
        hi = np.maximum(a1, a2)
        flat = lo * self.grid.n_bins + hi
        cols, inverse = np.unique(flat, return_inverse=True)
        values = np.bincount(inverse, weights=count).astype(np.float32)
        self.matrix.reshape(-1)[cols] += values  # `cols` is unique


def collect_band_stats(
        reader: ScoolReader, cells: Sequence[str],
        grids: Mapping[str, ResolutionGrid], n_strata: Mapping[str, int],
        max_cells: Optional[int] = None, random_state: int = 0,
        trans_res: Optional[str] = None
) -> "Tuple[Dict[str, BandStats], Optional[TransStats]]":
    r"""
    Stream over cells and accumulate band statistics at every resolution

    Parameters
    ----------
    reader
        Source ``.scool`` reader
    cells
        Cell names
    grids
        Bin grid per resolution
    n_strata
        Number of strata per resolution
    max_cells
        Number of cells to use, by default all of them
    random_state
        Random seed used when subsampling cells
    trans_res
        Resolution at which to also accumulate trans-chromosomal contacts

    Returns
    -------
    stats
        Band statistics per resolution
    trans
        Trans-chromosomal statistics, or ``None``
    """
    cells = list(cells)
    if max_cells and max_cells < len(cells):
        rs = np.random.RandomState(random_state)
        cells = [cells[i] for i in sorted(rs.choice(len(cells), max_cells, replace=False))]
    stats = {res: BandStats(grids[res], n_strata[res]) for res in grids}
    trans = TransStats(grids[trans_res]) if trans_res else None
    for cell in tqdm(cells, desc="Scanning cells"):
        try:
            bin1, bin2, count = reader.pixels(cell)
        except KeyError:
            continue
        for res, grid in grids.items():
            stats[res].update(*band_columns(grid, n_strata[res], bin1, bin2, count))
        if trans is not None:
            trans.update(bin1, bin2, count)
    return stats, trans


#----------------------------- Anchor selection --------------------------------

def select_anchors(
        stats: BandStats, max_anchors: Optional[int] = None,
        min_cells: int = 3, tile_size: int = 8,
        forced: Optional[np.ndarray] = None
) -> np.ndarray:
    r"""
    Choose which anchors of a resolution to keep

    Anchors are selected in contiguous tiles so that the retained band matrix
    keeps its local structure, which matters both for the convolutional encoder
    and for the stratum convolutions of the decoder.

    Parameters
    ----------
    stats
        Band statistics of the resolution
    max_anchors
        Maximal number of anchors to keep (``None`` or ``0`` keeps all
        detected anchors)
    min_cells
        Minimal number of cells in which an anchor must be detected
    tile_size
        Number of consecutive bins forming a selection tile
    forced
        Boolean mask of anchors to prioritize (typically bins overlapping
        promoters of highly variable genes). Prioritized tiles are still
        subject to ``max_anchors``.

    Returns
    -------
    anchors
        Sorted indices of the retained anchors
    """
    dispersion, cells = stats.anchor_scores()
    detected = cells >= min_cells
    if not detected.any():
        raise ValueError(
            f"No anchor of resolution '{stats.grid.name}' is detected in "
            f"at least {min_cells} cells!"
        )
    if not max_anchors or detected.sum() <= max_anchors:
        return np.where(detected)[0]

    tile_size = max(1, int(tile_size))
    chrom_code = stats.grid.chrom_code
    # Tiles never span chromosomes
    chrom_start = np.concatenate([[0], np.where(np.diff(chrom_code) != 0)[0] + 1])
    within = np.arange(chrom_code.size) - np.repeat(
        chrom_start, np.diff(np.append(chrom_start, chrom_code.size))
    )
    tile = chrom_code.astype(np.int64) * (chrom_code.size + 1) + within // tile_size
    tile = pd.factorize(tile)[0]
    n_tiles = tile.max() + 1

    tile_detected = np.bincount(tile, weights=detected, minlength=n_tiles)
    tile_score = np.bincount(
        tile, weights=np.where(detected, dispersion, 0.0), minlength=n_tiles
    ) / np.maximum(tile_detected, 1)
    tile_forced = np.zeros(n_tiles, dtype=bool)
    if forced is not None and np.any(forced):
        tile_forced = np.bincount(
            tile, weights=forced & detected, minlength=n_tiles
        ) > 0

    # Prioritized tiles first, then the most variable ones, until the budget
    # is exhausted
    order = np.lexsort((-tile_score, ~tile_forced))
    keep_tiles = np.zeros(n_tiles, dtype=bool)
    budget, dropped_forced = max_anchors, 0
    for t in order:
        if tile_detected[t] == 0:
            continue
        if tile_detected[t] > budget:
            dropped_forced += bool(tile_forced[t])
            continue
        keep_tiles[t] = True
        budget -= tile_detected[t]
    if dropped_forced:
        select_anchors_logger(
            f"Resolution '{stats.grid.name}': {dropped_forced} prioritized tiles "
            f"did not fit within the {max_anchors} anchor budget"
        )
    return np.where(detected & keep_tiles[tile])[0]


def select_anchors_logger(message: str) -> None:
    r"""
    Report anchor selection warnings
    """
    print(f"[multires] {message}")


def build_var(
        grid: ResolutionGrid, anchors: np.ndarray, n_strata: int
) -> pd.DataFrame:
    r"""
    Build the feature annotation of one resolution

    Features are laid out stratum-major over the retained anchors, i.e. exactly
    the band matrix the model reshapes them into.

    Parameters
    ----------
    grid
        Bin grid
    anchors
        Retained anchor indices
    n_strata
        Number of strata

    Returns
    -------
    var
        Feature annotation indexed by feature name
    """
    names = grid.names[anchors]
    bins = grid.bins.iloc[anchors].reset_index(drop=True)
    chrom_code = grid.chrom_code[anchors]
    # Target bin of every (anchor, stratum) pair, clipped to the chromosome
    n_bins = grid.n_bins
    frames = []
    for k in range(n_strata):
        target = np.minimum(anchors + k, n_bins - 1)
        same_chrom = grid.chrom_code[target] == chrom_code
        # Interactions running past the chromosome end fall back to the anchor
        target = np.where(same_chrom, target, anchors)
        frame = pd.DataFrame({
            "chrom": bins["chrom"].to_numpy(),
            "chromStart": bins["chromStart"].to_numpy(),
            "chromEnd": bins["chromEnd"].to_numpy(),
            "res": grid.name,
            "stratum": k,
            "anchor": names,
            "anchor_bin": anchors,
            "target": grid.names[target],
            "target_chrom": grid.bins["chrom"].to_numpy()[target],
            "target_chromStart": grid.bins["chromStart"].to_numpy()[target],
            "target_chromEnd": grid.bins["chromEnd"].to_numpy()[target],
            "in_chrom": same_chrom
        }, index=pd.Index(names if k == 0 else [f"{name}-{k}" for name in names]))
        frames.append(frame)
    var = pd.concat(frames)
    var.index.name = None
    return var


def mark_overlapping_bins(
        grid: ResolutionGrid, intervals: pd.DataFrame,
        chrom_key: str = "chrom", start_key: str = "chromStart",
        end_key: str = "chromEnd"
) -> np.ndarray:
    r"""
    Mark grid bins overlapping any of the given genomic intervals

    Parameters
    ----------
    grid
        Bin grid
    intervals
        Interval table
    chrom_key, start_key, end_key
        Column names of the interval table

    Returns
    -------
    mask
        Boolean mask over grid bins
    """
    mask = np.zeros(grid.n_bins, dtype=bool)
    bins = grid.bins
    starts = bins["chromStart"].to_numpy()
    ends = bins["chromEnd"].to_numpy()
    chrom_index = {}
    chrom_arr = bins["chrom"].to_numpy()
    change = np.where(chrom_arr[:-1] != chrom_arr[1:])[0] + 1
    bounds = np.concatenate([[0], change, [grid.n_bins]])
    for i in range(bounds.size - 1):
        chrom_index[chrom_arr[bounds[i]]] = (bounds[i], bounds[i + 1])
    for chrom, group in intervals.groupby(chrom_key, observed=True):
        if chrom not in chrom_index:
            continue
        lo, hi = chrom_index[chrom]
        left = np.searchsorted(ends[lo:hi], group[start_key].to_numpy(), side="right")
        right = np.searchsorted(starts[lo:hi], group[end_key].to_numpy(), side="left")
        for a, b in zip(left, right):
            if b > a:
                mask[lo + a:lo + b] = True
    return mask


def multires_suffix(
        resolutions: Sequence[str], n_strata: Sequence[int], loop_q,
        extra: str = ""
) -> str:
    r"""
    File name suffix identifying a multi-resolution preprocessing run
    """
    parts = [
        "multires",
        "-".join(str(res) for res in resolutions),
        "-".join(str(int(s)) for s in n_strata),
        str(loop_q)
    ]
    if extra:
        parts.append(str(extra))
    return "_".join(parts)


#-------------------------------- Graph building -------------------------------

def anchor_bed(grid: ResolutionGrid, anchors: np.ndarray) -> pd.DataFrame:
    r"""
    BED-style table of the retained anchors of one resolution
    """
    bed = grid.bins.iloc[anchors][
        ["chrom", "chromStart", "chromEnd", "name"]
    ].copy()
    bed.index = bed["name"]
    bed.index.name = None
    return bed


def loop_edges(
        stats: BandStats, anchors: np.ndarray, loop_q: float = 0.98,
        min_stratum: int = 1, coverage_norm: bool = False
) -> pd.DataFrame:
    r"""
    Top pseudobulk contacts of one resolution, as graph edges

    Contacts are ranked within each stratum, which makes the ranking
    observed/expected by construction.

    Parameters
    ----------
    stats
        Band statistics of the resolution
    anchors
        Retained anchor indices
    loop_q
        Quantile cutoff applied within each stratum
    min_stratum
        First stratum to consider (stratum 0 is the diagonal)
    coverage_norm
        Whether to additionally divide contacts by the square root of the
        product of their anchors' coverage (as in vanilla-coverage
        normalization), so that highly covered bins do not dominate

    Returns
    -------
    edges
        Edge table with ``source``, ``target``, ``weight`` and ``dist``
    """
    grid = stats.grid
    kept = np.zeros(grid.n_bins, dtype=bool)
    kept[anchors] = True
    names = grid.names
    starts = grid.bins["chromStart"].to_numpy()
    coverage = None
    if coverage_norm:
        coverage = np.sqrt(np.maximum(stats.anchor_sum, 0.0))
        coverage[coverage == 0] = 1.0
    frames = []
    for k in range(max(1, min_stratum), stats.n_strata):
        target = anchors + k
        valid = target < grid.n_bins
        source_k, target_k = anchors[valid], target[valid]
        valid = kept[target_k] & (grid.chrom_code[source_k] == grid.chrom_code[target_k])
        source_k, target_k = source_k[valid], target_k[valid]
        if not source_k.size:
            continue
        value = stats.feature_sum[k * grid.n_bins + source_k]
        nonzero = value > 0
        source_k, target_k, value = source_k[nonzero], target_k[nonzero], value[nonzero]
        if not source_k.size:
            continue
        if coverage is not None:
            value = value / (coverage[source_k] * coverage[target_k])
        rank = pd.Series(value).rank(pct=True).to_numpy()
        keep = rank >= np.quantile(rank, loop_q) if loop_q > 0 else np.ones_like(rank, dtype=bool)
        if not keep.any():
            continue
        source_k, target_k = source_k[keep], target_k[keep]
        weight = pd.Series(rank[keep]).rank(pct=True).to_numpy()
        frames.append(pd.DataFrame({
            "source": names[source_k],
            "target": names[target_k],
            "weight": weight,
            "dist": np.abs(
                starts[target_k].astype(np.float64) - starts[source_k]
            ) / 2e6
        }))
    if not frames:
        return pd.DataFrame(columns=["source", "target", "weight", "dist"])
    return pd.concat(frames, ignore_index=True)


def trans_edges(
        stats: TransStats, anchors: np.ndarray, loop_q: float = 0.98,
        coverage_norm: bool = False
) -> pd.DataFrame:
    r"""
    Top pseudobulk trans-chromosomal contacts, as graph edges

    The cutoff is relaxed per chromosome pair the same way the
    single-resolution pipeline does it, so that the total number of trans edges
    is comparable to the number of cis edges kept at ``loop_q``.

    Parameters
    ----------
    stats
        Trans-chromosomal statistics
    anchors
        Retained anchor indices of the same resolution
    loop_q
        Cis quantile cutoff the trans cutoff is derived from
    coverage_norm
        Whether to divide contacts by the square root of the product of their
        anchors' trans coverage

    Returns
    -------
    edges
        Edge table with ``source``, ``target``, ``weight`` and ``dist``
    """
    grid = stats.grid
    chrom_code = grid.chrom_code
    n_chrom = int(chrom_code.max()) + 1
    if n_chrom < 2:
        return pd.DataFrame(columns=["source", "target", "weight", "dist"])
    n_pairs = n_chrom * (n_chrom - 1) / 2
    trans_q = max(0.0, 1 - (1 - loop_q) / n_pairs)

    matrix = stats.matrix[np.ix_(anchors, anchors)]
    matrix = matrix + matrix.T  # only one triangle was accumulated
    codes = chrom_code[anchors]
    names = grid.names[anchors]
    coverage = None
    if coverage_norm:
        coverage = np.sqrt(np.maximum(matrix.sum(axis=1), 0.0))
        coverage[coverage == 0] = 1.0
    frames = []
    for chrom in range(n_chrom):
        rows = np.where(codes == chrom)[0]
        cols = np.where(codes != chrom)[0]
        if not rows.size or not cols.size:
            continue
        block = matrix[np.ix_(rows, cols)]
        source_i, target_i = np.nonzero(block)
        if not source_i.size:
            continue
        value = block[source_i, target_i]
        source_i, target_i = rows[source_i], cols[target_i]
        if coverage is not None:
            value = value / (coverage[source_i] * coverage[target_i])
        rank = pd.Series(value).rank(pct=True).to_numpy()
        keep = rank >= np.quantile(rank, trans_q)
        if not keep.any():
            continue
        frames.append(pd.DataFrame({
            "source": names[source_i[keep]],
            "target": names[target_i[keep]],
            "weight": pd.Series(rank[keep]).rank(pct=True).to_numpy(),
            "dist": 1.0
        }))
    if not frames:
        return pd.DataFrame(columns=["source", "target", "weight", "dist"])
    return pd.concat(frames, ignore_index=True)


def adjacency_edges(grid: ResolutionGrid, anchors: np.ndarray) -> pd.DataFrame:
    r"""
    Edges between consecutive retained anchors of the same chromosome

    These keep the genome connected even where no contact passes the loop
    cutoff, mirroring the sequential edges of the single-resolution pipeline.
    """
    if anchors.size < 2:
        return pd.DataFrame(columns=["source", "target", "weight", "dist"])
    source, target = anchors[:-1], anchors[1:]
    same = grid.chrom_code[source] == grid.chrom_code[target]
    source, target = source[same], target[same]
    starts = grid.bins["chromStart"].to_numpy()
    return pd.DataFrame({
        "source": grid.names[source],
        "target": grid.names[target],
        "weight": 1.0,
        "dist": np.abs(
            starts[target].astype(np.float64) - starts[source]
        ) / 2e6
    })


def hierarchy_edges(
        fine: ResolutionGrid, fine_anchors: np.ndarray,
        coarse: ResolutionGrid, coarse_anchors: np.ndarray
) -> pd.DataFrame:
    r"""
    Edges linking every retained fine anchor to the coarse anchor containing it

    Both grids are derived from the same source bins, so containment is read
    off the two source-bin mappings. These edges are what makes the guidance
    graph a genuine multi-scale graph: information (and reachability from
    genes) propagates from coarse bins down to high resolution anchors.
    """
    fine_pos = np.full(fine.n_bins, -1, dtype=np.int64)
    fine_pos[fine_anchors] = 1
    coarse_pos = np.full(coarse.n_bins, -1, dtype=np.int64)
    coarse_pos[coarse_anchors] = 1
    fine_bin, coarse_bin = fine.base_to_bin, coarse.base_to_bin
    valid = (fine_bin >= 0) & (coarse_bin >= 0)
    fine_bin, coarse_bin = fine_bin[valid], coarse_bin[valid]
    valid = (fine_pos[fine_bin] > 0) & (coarse_pos[coarse_bin] > 0)
    fine_bin, coarse_bin = fine_bin[valid], coarse_bin[valid]
    if not fine_bin.size:
        return pd.DataFrame(columns=["source", "target", "weight", "dist"])
    pairs = np.unique(np.stack([fine_bin, coarse_bin], axis=1), axis=0)
    return pd.DataFrame({
        "source": fine.names[pairs[:, 0]],
        "target": coarse.names[pairs[:, 1]],
        "weight": 1.0,
        "dist": 0.0
    })


#------------------------------ Incremental h5ad -------------------------------

class SparseH5adWriter:

    r"""
    Write an :class:`anndata.AnnData` object with a sparse ``X`` one chunk of
    cells at a time.

    The whole point is to never hold the full matrix (or a dense minibatch of
    all features) in memory, so that high resolution datasets can be written
    from a single streaming pass.

    Parameters
    ----------
    path
        Output path
    n_vars
        Number of features
    dtype
        Dtype of ``X``
    """

    def __init__(
            self, path: os.PathLike, n_vars: int, dtype: type = np.float32
    ) -> None:
        self.path = str(path)
        self.n_vars = int(n_vars)
        self.dtype = np.dtype(dtype)
        self.n_obs = 0
        self.handle = h5py.File(self.path, "w")
        self.handle.attrs["encoding-type"] = "anndata"
        self.handle.attrs["encoding-version"] = "0.1.0"
        group = self.handle.create_group("X")
        group.attrs["encoding-type"] = "csr_matrix"
        group.attrs["encoding-version"] = "0.1.0"
        self._data = group.create_dataset(
            "data", shape=(0, ), maxshape=(None, ), dtype=self.dtype,
            chunks=(2 ** 16, ), compression="gzip", compression_opts=1
        )
        # `indices` and `indptr` must share a dtype for scipy to accept the
        # matrix, and int64 keeps very large datasets (nnz > 2^31) writable
        self._indices = group.create_dataset(
            "indices", shape=(0, ), maxshape=(None, ), dtype=np.int64,
            chunks=(2 ** 16, ), compression="gzip", compression_opts=1
        )
        self._indptr = [0]

    def write_chunk(
            self, entries: "List[Tuple[np.ndarray, np.ndarray]]"
    ) -> None:
        r"""
        Append a chunk of cells given as ``(columns, values)`` pairs
        """
        sizes = [columns.size for columns, _ in entries]
        rows = scipy.sparse.csr_matrix(
            (
                np.concatenate([values for _, values in entries])
                if entries else np.empty(0, dtype=self.dtype),
                np.concatenate([columns for columns, _ in entries])
                if entries else np.empty(0, dtype=np.int64),
                np.concatenate([[0], np.cumsum(sizes)])
            ),
            shape=(len(entries), self.n_vars)
        )
        self.write_rows(rows)

    def write_rows(self, rows: scipy.sparse.csr_matrix) -> None:
        r"""
        Append a chunk of cells
        """
        rows = rows.tocsr()
        if rows.shape[1] != self.n_vars:
            raise ValueError(
                f"Expected {self.n_vars} features, got {rows.shape[1]}!"
            )
        rows.sort_indices()
        nnz = self._data.shape[0]
        self._data.resize((nnz + rows.data.size, ))
        self._data[nnz:] = rows.data.astype(self.dtype)
        self._indices.resize((nnz + rows.indices.size, ))
        self._indices[nnz:] = rows.indices.astype(np.int64)
        self._indptr += (rows.indptr[1:] + nnz).tolist()
        self.n_obs += rows.shape[0]

    def finalize(
            self, obs: pd.DataFrame, var: pd.DataFrame,
            uns: Optional[Mapping] = None
    ) -> None:
        r"""
        Write annotations and close the file
        """
        try:  # anndata >= 0.11
            from anndata.io import write_elem
        except ImportError:
            from anndata._io.specs import write_elem

        if obs.shape[0] != self.n_obs:
            raise ValueError(
                f"`obs` has {obs.shape[0]} rows but {self.n_obs} cells were written!"
            )
        if var.shape[0] != self.n_vars:
            raise ValueError(
                f"`var` has {var.shape[0]} rows but X has {self.n_vars} columns!"
            )
        group = self.handle["X"]
        group.attrs["shape"] = np.array([self.n_obs, self.n_vars], dtype=np.int64)
        group.create_dataset(
            "indptr", data=np.asarray(self._indptr, dtype=np.int64)
        )
        write_elem(self.handle, "obs", obs)
        write_elem(self.handle, "var", var)
        write_elem(self.handle, "uns", dict(uns or {}))
        for key in ("obsm", "varm", "layers", "obsp", "varp"):
            write_elem(self.handle, key, {})
        self.handle.close()

    def close(self) -> None:
        r"""
        Close the file without writing annotations
        """
        try:
            self.handle.close()
        except Exception:  # pylint: disable=broad-except
            pass


#-------------------------------- Feature layout -------------------------------

class MultiResLayout:

    r"""
    Feature layout of a multi-resolution Hi-C dataset

    Parameters
    ----------
    grids
        Bin grid per resolution
    anchors
        Retained anchor indices per resolution
    n_strata
        Number of strata per resolution
    res_order
        Resolution order, coarse to fine
    """

    def __init__(
            self, grids: Mapping[str, ResolutionGrid],
            anchors: Mapping[str, np.ndarray],
            n_strata: Mapping[str, int],
            res_order: Sequence[str]
    ) -> None:
        self.res_order = list(res_order)
        self.grids = grids
        self.anchors = {res: np.asarray(anchors[res]) for res in self.res_order}
        self.n_strata = {res: int(n_strata[res]) for res in self.res_order}
        self.offset, self.anchor_pos = {}, {}
        offset = 0
        for res in self.res_order:
            self.offset[res] = offset
            pos = np.full(grids[res].n_bins, -1, dtype=np.int64)
            pos[self.anchors[res]] = np.arange(self.anchors[res].size)
            self.anchor_pos[res] = pos
            offset += self.anchors[res].size * self.n_strata[res]
        self.n_features = offset

    def var(self) -> pd.DataFrame:
        r"""
        Feature annotation of all resolutions, in feature order

        Raises
        ------
        ValueError
            If two resolutions produce the same feature name. Feature names
            identify both model features and graph vertices, so a collision
            would silently merge them.
        """
        var = pd.concat([
            build_var(self.grids[res], self.anchors[res], self.n_strata[res])
            for res in self.res_order
        ])
        if not var.index.is_unique:
            duplicated = var.index[var.index.duplicated()].unique()
            raise ValueError(
                f"{duplicated.size} feature names are shared by more than one "
                f"resolution (e.g. {list(duplicated[:3])}). Bin names must be "
                f"unique across resolutions."
            )
        return var

    def row(
            self, bin1: np.ndarray, bin2: np.ndarray, count: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Convert the pixels of one cell into sparse row entries

        Parameters
        ----------
        bin1, bin2, count
            Pixel table of one cell, in source bin indices

        Returns
        -------
        columns
            Feature columns
        values
            Contact counts
        """
        columns, values = [], []
        for res in self.res_order:
            grid, n_strata = self.grids[res], self.n_strata[res]
            cols, vals = band_columns(grid, n_strata, bin1, bin2, count)
            if not cols.size:
                continue
            anchor = self.anchor_pos[res][cols % grid.n_bins]
            keep = anchor >= 0
            if not keep.any():
                continue
            stratum = cols[keep] // grid.n_bins
            n_anchors = self.anchors[res].size
            columns.append(
                self.offset[res] + stratum * n_anchors + anchor[keep]
            )
            values.append(vals[keep])
        if not columns:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        return np.concatenate(columns), np.concatenate(values)


def write_multires_hic(
        path: os.PathLike, reader: ScoolReader, cells: Sequence[str],
        layout: MultiResLayout, obs: pd.DataFrame,
        chunk_size: int = 256, uns: Optional[Mapping] = None
) -> "Dict[str, np.ndarray]":
    r"""
    Stream all cells into a sparse multi-resolution ``.h5ad`` file

    Parameters
    ----------
    path
        Output path
    reader
        Source ``.scool`` reader
    cells
        Cell names, in the same order as ``obs``
    layout
        Feature layout
    obs
        Cell annotation
    chunk_size
        Number of cells buffered before writing
    uns
        Unstructured annotation

    Returns
    -------
    depths
        Per-resolution in-band contact counts of every cell
    """
    writer = SparseH5adWriter(path, layout.n_features)
    res_of_feature = np.concatenate([
        np.full(layout.anchors[res].size * layout.n_strata[res], i, dtype=np.int64)
        for i, res in enumerate(layout.res_order)
    ])
    obs = obs.copy()
    depths = np.zeros((len(cells), len(layout.res_order)), dtype=np.float64)
    try:
        buffer, buffer_rows = [], 0
        for i, cell in enumerate(tqdm(cells, desc="Writing cells")):
            try:
                bin1, bin2, count = reader.pixels(cell)
                columns, values = layout.row(bin1, bin2, count)
            except KeyError:
                columns = np.empty(0, dtype=np.int64)
                values = np.empty(0, dtype=np.float32)
            depths[i] = np.bincount(
                res_of_feature[columns], weights=values,
                minlength=len(layout.res_order)
            )
            buffer.append((columns, values))
            buffer_rows += 1
            if buffer_rows >= chunk_size:
                writer.write_chunk(buffer)
                buffer, buffer_rows = [], 0
        if buffer:
            writer.write_chunk(buffer)
        for i, res in enumerate(layout.res_order):
            obs[f"depth_{res}"] = depths[:, i]
        writer.finalize(obs, layout.var(), uns=uns)
    except Exception:
        writer.close()
        raise
    return {
        res: depths[:, i] for i, res in enumerate(layout.res_order)
    }
