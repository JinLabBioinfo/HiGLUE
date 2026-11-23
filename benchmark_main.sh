#!/usr/bin/env bash
# Main results of the HiGLUE paper: every dataset of datasets/ trained with the
# default configuration of higlue.py (as in hires_minimal.sh, with O/E-ranked
# contact edges) over several seeds, each run followed by the evaluation
# (evals.py), the baselines on the same cells (scGAD + Harmony, scgad.py; Hi-C
# only clusters named through the guidance graph, graph_match.py) and,
# for the islets, the pseudobulk structure analysis (structure_eval.py).
#
# Inputs come only from the dataset tree (datasets/link_datasets.sh locally,
# datasets/sync_to_cluster.sh on the cluster), so the script runs unchanged in
# both places. Ablations live in benchmark.sh and islet_benchmark/sweep.sh.
#
# Usage:
#   ./benchmark_main.sh --list                       # datasets and their settings
#   ./benchmark_main.sh --dry_run                    # print every command
#   ./benchmark_main.sh --prep_only                  # preprocess all datasets
#   ./benchmark_main.sh                              # preprocess (if needed) and train
#   ./benchmark_main.sh --datasets islet_atlas,islet_inhouse --seeds "1 2 3 4 5"
#   EXTRA_ARGS="--lam_align 0.002" ./benchmark_main.sh --datasets islet_atlas   # other options
#   TRAIN_ARGS="--decay_norm cell" ./benchmark_main.sh --datasets islet_inhouse   # + per-cell distance decay
#   TRAIN_ARGS="--comp_weight none" ./benchmark_main.sh                       # ablation: no composition weights
#   TRAIN_ARGS="--no-gad_modality --comp_weight none" ./benchmark_main.sh     # ablation: Hi-C only model
#   ./benchmark_main.sh --res "500kb 100kb"          # multi-resolution model
#   ./benchmark_main.sh --shard 0/4                  # 1st of 4 parallel workers
#   python benchmark_summary.py                      # tables and main_comparison.png of all finished runs
#
# Options (also settable as environment variables of the same name, upper case):
#   --datasets a,b       subset of datasets (default: all)
#   --seeds "1 2 3"      training seeds
#   --res "R1 R2"        model resolution (default 500kb; the dataset needs hic_<res>.scool).
#                        Several resolutions train the multi-resolution model, which can
#                        also start from the closest finer scool
#   EXTRA_ARGS="..."     higlue.py options added to every command (environment only); each
#                        value gets its own preprocessing and run directories
#   TRAIN_ARGS="..."     higlue.py options added to training only (environment only), e.g.
#                        TRAIN_ARGS="--decay_norm cell"; shares the preprocessing, each value
#                        gets its own run directories (<res>_<config>_<model>-train-<hash>)
#   RUNTIME_ARGS="..."   higlue.py training options that do not change the model (figures,
#                        logging), e.g. "--save_interval 100 --viz none"; not part of the
#                        run directory name, so runs with and without them are the same run
#   MODEL_TAG=name       names the model defaults in the run directories (default gadcw:
#                        gene-body modality + graph-estimated composition weights, the
#                        higlue.py defaults since they were introduced)
#   --baselines "scgad scgad_cca graph_match"  baselines run after training ("" for none)
#   --max_epochs N       cap training epochs (default: early stopping)
#   --backed             read the Hi-C data from disk during training
#   --shard I/N          run only every N-th training run, starting at I (preprocessing
#                        must exist: run --prep_only first)
#   --prep_only          only preprocess
#   --force_preprocess   redo preprocessing
#   --skip_done          skip runs that already finished (default: rerun)
#   --no_wandb           do not log to wandb
#   --dry_run            print commands only
#   --data_root DIR      dataset tree (datasets)
#   --prep_root DIR      preprocessed data (main_data)
#   --run_root DIR       runs (main_runs)
#
# Outputs:
#   main_data/<dataset>/<res>_<config>/           preprocessed data, shared by all seeds
#                                                 (<config> = default, or a hash of EXTRA_ARGS)
#   main_runs/<dataset>/<res>_<config>_<model>/seed<N>/train.log   log of one run
#   main_runs/<dataset>/<res>_<config>/seed<N>/eval/       HiGLUE metrics, plots, embeddings
#                         .../eval/baselines/scgad/  scGAD + Harmony on the same cells
#                         .../eval/baselines/graph_match/  graph matching (clusters, composition, UMAP)
#                         .../eval/comp_weight/    composition estimate and RNA weights (--comp_weight graph)
#                         .../eval/structure/      pseudobulk structure (islets)
#   main_runs/<dataset>/<res>_<config>/seed<N>/model/      trained model and predictions
#   main_runs/summary.tsv                         status and runtime of every run

set -uo pipefail
cd "$(dirname "$0")"

PYTHON=${PYTHON:-python}
DATASETS=${DATASETS:-}
SEEDS=${SEEDS:-"1 2 3"}
RES=${RES:-500kb}
BASELINES=${BASELINES-"scgad scgad_cca graph_match"}
MAX_EPOCHS=${MAX_EPOCHS:-}
BACKED=${BACKED:-0}
SHARD=${SHARD:-}
PREP_ONLY=${PREP_ONLY:-0}
FORCE_PREPROCESS=${FORCE_PREPROCESS:-0}
SKIP_DONE=${SKIP_DONE:-0}
USE_WANDB=${USE_WANDB:-1}
STRUCTURE_EVAL=${STRUCTURE_EVAL:-1}  # 0: skip the pseudobulk structure analysis of the islet datasets (ablations)
DRY_RUN=${DRY_RUN:-0}
DATA_ROOT=${DATA_ROOT:-datasets}
PREP_ROOT=${PREP_ROOT:-main_data}
RUN_ROOT=${RUN_ROOT:-main_runs}
# appended to dataset names, so main runs get their own wandb projects
# (GLUE-<dset><suffix>-hic-2d-prior)
DSET_SUFFIX=${DSET_SUFFIX:-_main}

# Model configuration: the defaults of higlue.py as in hires_minimal.sh (single
# resolution, composition decoder, 10,000 genes, 10 strata, q = 0.98, batch size
# 128, ...), except that contact edges are ranked by observed/expected rather
# than raw pseudobulk counts (raw counts only select near-diagonal pixels).
# Only data filters differ between datasets. Extra higlue.py options for every
# run can be passed with EXTRA_ARGS="--opt value ...".
# genes are linked to the bins overlapping their bodies (as in the original HiGLUE)
BASE_ARGS=(--use_dist_norm --gene_edges body)
EXTRA_ARGS=${EXTRA_ARGS:-}
TRAIN_ARGS=${TRAIN_ARGS:-}
RUNTIME_ARGS=${RUNTIME_ARGS:-}

ALL_DATASETS=(pfc human_brain hires_brain droplet_cortex islet_atlas islet_inhouse gageseq_cortex gageseq_bm hires_embryo hires_embryo_full hires_embryo_full_nomito)

# ---------------------------------------------------------------------------
# Datasets: cell filters and the few settings that differ between datasets.
# Dataset names matter: higlue.py restricts the PFC and human brain RNA to the
# Hi-C cell types by name, and evals.py picks the islet label map by name.
# ---------------------------------------------------------------------------
dataset_config() {
    DSET=$1
    MIN_DEPTH=0
    MAX_DEPTH=""
    GENE_LIST=()
    SCORE_FLAGS=()    # SCORE options (after SCORE): data filters only
    STRUCTURE=0       # pseudobulk structure analysis (needs hic_100kb.scool)
    SCGAD_RES=5kb     # scool used by scGAD
    case $1 in
        pfc|pfc_full)   # pfc_full: all 14 cell types (pfc: neurons only, see higlue.py)
            MIN_DEPTH=50000
            SCGAD_RES=50kb      # finest PFC scool (datasets/build_pfc_scools.py)
            GENE_LIST=(NDNF SV2C GAD1 PVALB SST VIP CUX2 RORB FEZF2 FOXP2) ;;
        human_brain)
            MIN_DEPTH=200000
            GENE_LIST=(SYT1 FOXP2 LAMP5 PBX3 LSAMP NR2F1 TSHZ2 INPP5J KCNQ5 CNTN4 FAM172A)
            SCGAD_RES=100kb ;;  # the only resolution of the neuron scool
        hires_brain)
            MIN_DEPTH=100000
            GENE_LIST=(Rorb Schip1 Zfpm2 Tle4 Nxph3 Tshz2 Meg3 Scn4b Meis2 Gad2 S1pr1 Slc1a3 Cldn11 Mbp) ;;
        hires_embryo|hires_embryo_full|hires_embryo_full_nomito)   # HiRES mouse embryo (paired): the EX15 cells
            # (1,122 Hi-C, 1,121 RNA), or all seven stages E7.0-EX15 (7,469 Hi-C, 7,464 RNA;
            # datasets/build_hires_embryo_full.py; 7,160 >= 10^5 cis); _nomito: without mitotic cells
            # (embryo_benchmark/make_nomito.py; 6,735 Hi-C >= 10^5 cis, 6,906 RNA)
            MIN_DEPTH=100000
            GENE_LIST=(Col1a1 Twist1 Tubb3 Neurod1 Hba-a1 Sox2 Pax6 Olig2 Myod1 Epcam Foxa2 Sox10) ;;
        droplet_cortex|droplet_snare)   # droplet_snare: the same Hi-C with SNARE-seq RNA (+ ATAC in EXTRA_ARGS)
            MIN_DEPTH=1000
            GENE_LIST=(Rorb Schip1 Zfpm2 Tle4 Nxph3 Tshz2 Meg3 Scn4b Meis2 Gad2 S1pr1 Slc1a3 Cldn11 Mbp)
            SCORE_FLAGS=(--ignore_chr_filter) ;;
        islet_atlas)
            MIN_DEPTH=5000
            GENE_LIST=(INS GCG SST PPY GHRL HNF1A HNF1B RFX6 ARX IRX2 MAFA PDX1 NKX6-1 FXYD2 PPP1R1A CDKN2A)
            STRUCTURE=1 ;;
        gageseq_cortex)   # GAGE-seq mouse cortex (paired; datasets/build_gageseq.py)
            MIN_DEPTH=50000
            GENE_LIST=(Rorb Schip1 Zfpm2 Tle4 Nxph3 Tshz2 Meg3 Scn4b Meis2 Gad2 S1pr1 Slc1a3 Cldn11 Mbp) ;;
        gageseq_bm)       # GAGE-seq human bone marrow CD34+ (paired)
            MIN_DEPTH=20000
            GENE_LIST=(CD34 CRHBP HLF AVP MPO DNTT GATA1 KLF1 CD79A IL7R) ;;
        islet_inhouse|islet_full)   # islet_full: islet_inhouse + the y19 donor (scHi-C/islet/build_islet_full.sh)
            MIN_DEPTH=50000
            MAX_DEPTH=1500000
            GENE_LIST=(INS GCG SST PPY GHRL HNF1A HNF1B RFX6 ARX IRX2 MAFA PDX1 NKX6-1 FXYD2 PPP1R1A CDKN2A)
            STRUCTURE=1 ;;
        *) echo "Unknown dataset: $1 (have: ${ALL_DATASETS[*]} pfc_full droplet_snare islet_full)" >&2; return 1 ;;
    esac
    # --min_depth in EXTRA_ARGS replaces the dataset's depth cutoff (higlue.py and SCORE parse one --min_depth; the
    # SCORE value comes last and wins) and, as part of EXTRA_ARGS, gets its own preprocessed data directory
    if [[ " $EXTRA_ARGS " =~ \ --min_depth[=\ ]+([0-9]+) ]]; then MIN_DEPTH=${BASH_REMATCH[1]}; fi
    DIR=$DATA_ROOT/$1
}

res_to_bp() {
    local res=${1,,}
    case $res in
        *mb) echo $(( ${res%mb} * 1000000 )) ;;
        *kb) echo $(( ${res%kb} * 1000 )) ;;
        *bp) echo "${res%bp}" ;;
        *) echo "$res" ;;
    esac
}

pick_scool() {
    # the dataset's scool at the finest model resolution; a multi-resolution
    # model can also start from the coarsest available finer one (higlue.py
    # derives coarser grids from it), the default model cannot
    local finest=$1 best="" best_bp=0 f r bp
    if [[ -f $DIR/hic_${finest}.scool ]]; then
        SCOOL_RES=$finest; return 0
    fi
    (( $(wc -w <<< "$RES") > 1 )) || return 1
    for f in "$DIR"/hic_*.scool; do
        [[ -e $f ]] || continue
        r=${f##*/hic_}; r=${r%.scool}
        bp=$(res_to_bp "$r")
        if (( bp < $(res_to_bp "$finest") && bp > best_bp )); then best=$r; best_bp=$bp; fi
    done
    [[ -n $best ]] || return 1
    SCOOL_RES=$best
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --datasets) DATASETS=$2; shift 2 ;;
        --seeds) SEEDS=$2; shift 2 ;;
        --res) RES=$2; shift 2 ;;
        --baselines) BASELINES=$2; shift 2 ;;
        --max_epochs) MAX_EPOCHS=$2; shift 2 ;;
        --backed) BACKED=1; shift ;;
        --shard) SHARD=$2; shift 2 ;;
        --prep_only) PREP_ONLY=1; shift ;;
        --force_preprocess) FORCE_PREPROCESS=1; shift ;;
        --skip_done) SKIP_DONE=1; shift ;;
        --no_wandb) USE_WANDB=0; shift ;;
        --dry_run) DRY_RUN=1; shift ;;
        --data_root) DATA_ROOT=$2; shift 2 ;;
        --prep_root) PREP_ROOT=$2; shift 2 ;;
        --run_root) RUN_ROOT=$2; shift 2 ;;
        --list)
            for d in "${ALL_DATASETS[@]}"; do
                dataset_config "$d"
                printf '%-15s min_depth %-7s max_depth %-8s structure %s scgad %-6s %s\n' \
                    "$d" "$MIN_DEPTH" "${MAX_DEPTH:--}" "$STRUCTURE" "$SCGAD_RES" "${SCORE_FLAGS[*]}"
            done
            exit 0 ;;
        -h|--help) sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done
FINEST=""
for r in $RES; do
    if [[ -z $FINEST ]] || (( $(res_to_bp "$r") < $(res_to_bp "$FINEST") )); then FINEST=$r; fi
done
RES_TAG=$(echo "$RES" | xargs | tr ' ' '-')
# preprocessed file names do not record every setting, so each configuration
# gets its own preprocessing (and run) directory
# (the earlier promoter-linked default lives in *_default directories)
CONFIG_TAG=genebody
[[ -n $EXTRA_ARGS ]] && CONFIG_TAG=genebody-args-$(echo "$EXTRA_ARGS" | md5sum | cut -c1-8)
RES_TAG=${RES_TAG}_${CONFIG_TAG}
# training-only options leave the preprocessing alone; runs of the current
# model defaults are kept apart from runs of earlier defaults by MODEL_TAG
MODEL_TAG=${MODEL_TAG:-gadcw}
RUN_TAG=${RES_TAG}_${MODEL_TAG}
[[ -n $TRAIN_ARGS ]] && RUN_TAG=${RUN_TAG}-train-$(echo "$TRAIN_ARGS" | md5sum | cut -c1-8)

build_cmd() {
    # $1 = preprocess | train, $2 = seed, $3 = run directory
    local stage=$1 seed=$2 run_dir=$3
    CMD=(
        "$PYTHON" higlue.py
        --rna_file "$DIR/rna.h5ad"
        --gtf "$DIR/genes.gtf"
        --data_dir "$DATA_DIR"
        --seed "$seed"
    )
    # several resolutions = the multi-resolution model; one = the default model
    (( $(wc -w <<< "$RES") > 1 )) && CMD+=(--resolutions $RES)
    CMD+=("${BASE_ARGS[@]}" $EXTRA_ARGS)
    if [[ $stage == preprocess ]]; then
        CMD+=(--preprocess)
    else
        CMD+=(--train --exp_name "$run_dir/glue" --cache_checkpoint "$run_dir/model"
              --eval_dir "$run_dir/eval" $TRAIN_ARGS $RUNTIME_ARGS)
        # genes for the figures and predicted expression only; passed at training
        # so that they do not change the preprocessed gene set
        (( ${#GENE_LIST[@]} )) && CMD+=(--gene_list "${GENE_LIST[@]}")
        (( BACKED )) && CMD+=(--backed)
        [[ -n $MAX_EPOCHS ]] && CMD+=(--max_epochs "$MAX_EPOCHS")
        (( USE_WANDB )) && CMD+=(--wandb)
        [[ -n $BASELINES ]] && CMD+=(--baselines $BASELINES)
        # scGAD and the gene-body modality (on by default) count at SCGAD_RES
        CMD+=(--scgad_scool "$DIR/hic_${SCGAD_RES}.scool")
        (( STRUCTURE && STRUCTURE_EVAL )) && CMD+=(--structure_eval --structure_scool "$DIR/hic_100kb.scool")
    fi
    CMD+=(
        SCORE
        --subname glue
        --dset "${DSET}${DSET_SUFFIX}"
        --scool "$DIR/hic_${SCOOL_RES}.scool"
        --reference "$DIR/hic_ref.tsv"
        --resolution "$SCOOL_RES"
        --min_depth "$MIN_DEPTH"
        "${SCORE_FLAGS[@]}"
    )
    [[ -n $MAX_DEPTH ]] && CMD+=(--max_depth "$MAX_DEPTH")
    return 0
}

run_logged() {
    # $1 = log file, rest = command; prints the command, then runs it (tee'd)
    local log=$1; shift
    printf "+"; printf " %q" "$@"; echo
    (( DRY_RUN )) && return 0
    mkdir -p "$(dirname "$log")"
    { printf "+"; printf " %q" "$@"; echo; } > "$log"
    "$@" 2>&1 | tee -a "$log"
}

IFS=',' read -r -a datasets <<< "${DATASETS:-$(IFS=,; echo "${ALL_DATASETS[*]}")}"
shard_i=0; shard_n=1
if [[ -n $SHARD ]]; then
    shard_i=${SHARD%/*}; shard_n=${SHARD#*/}
fi

# check inputs first, so a long run does not stop at the 5th dataset
n_failed=0
ok_datasets=()
for d in "${datasets[@]}"; do
    dataset_config "$d" || exit 1
    missing=()
    for f in rna.h5ad genes.gtf hic_ref.tsv; do [[ -e $DIR/$f ]] || missing+=("$f"); done
    pick_scool "$FINEST" || missing+=("hic_<=${FINEST}.scool")
    [[ -e $DIR/hic_${SCGAD_RES}.scool ]] || missing+=("hic_${SCGAD_RES}.scool")
    (( STRUCTURE )) && [[ ! -e $DIR/hic_100kb.scool ]] && missing+=("hic_100kb.scool")
    if (( ${#missing[@]} )); then
        echo "!! $d: missing ${missing[*]} in $DIR, skipping (run datasets/link_datasets.sh; on the cluster: purged from /scratch? copy the bundle again)" >&2
        n_failed=$((n_failed + 1))
    else
        ok_datasets+=("$d")
    fi
done

SUMMARY=$RUN_ROOT/summary.tsv
if (( ! DRY_RUN )); then
    mkdir -p "$RUN_ROOT"
    [[ -f $SUMMARY ]] || printf 'date\tdataset\tresolutions\tseed\tstatus\tseconds\tlog\n' > "$SUMMARY"
fi

echo "Datasets: ${ok_datasets[*]}"
echo "Resolutions: $RES   Seeds: $SEEDS   Baselines: ${BASELINES:-none}   Shard: ${SHARD:-all}"

run_index=0
for d in "${ok_datasets[@]}"; do
    dataset_config "$d"
    pick_scool "$FINEST"
    DATA_DIR=$PREP_ROOT/$d/$RES_TAG
    echo
    echo "=== $d / $RES_TAG (input ${SCOOL_RES} scool, data: $DATA_DIR)"

    if (( FORCE_PREPROCESS )) || [[ ! -f $DATA_DIR/.preprocessed ]]; then
        if [[ -n $SHARD ]] && (( ! PREP_ONLY )); then
            echo "!! $d is not preprocessed; run --prep_only before sharded training" >&2
            n_failed=$((n_failed + 1))
            continue
        fi
        build_cmd preprocess 0 ""
        if run_logged "$DATA_DIR/preprocess.log" "${CMD[@]}"; then
            (( DRY_RUN )) || touch "$DATA_DIR/.preprocessed"
        else
            echo "!! preprocessing failed, see $DATA_DIR/preprocess.log" >&2
            (( DRY_RUN )) || printf '%s\t%s\t%s\t-\tpreprocess_failed\t-\t%s\n' \
                "$(date -Is)" "$d" "$RES_TAG" "$DATA_DIR/preprocess.log" >> "$SUMMARY"
            n_failed=$((n_failed + 1))
            continue
        fi
    fi
    (( PREP_ONLY )) && continue

    for seed in $SEEDS; do
        this_run=$run_index
        run_index=$((run_index + 1))
        (( this_run % shard_n == shard_i )) || continue
        run_dir=$RUN_ROOT/$d/$RUN_TAG/seed$seed
        if (( SKIP_DONE )) && [[ -f $run_dir/.done ]]; then
            echo "--- seed $seed already done, skipping"
            continue
        fi
        build_cmd train "$seed" "$run_dir"
        export WANDB_NAME="${d}_${RUN_TAG}_s${seed}"
        export WANDB_RUN_GROUP="main_${RUN_TAG}"
        export WANDB_TAGS="main,${RUN_TAG}"
        rm -f "$run_dir/.done"
        start=$SECONDS
        if run_logged "$run_dir/train.log" "${CMD[@]}"; then
            status=ok
            (( DRY_RUN )) || touch "$run_dir/.done"
        else
            status=failed
            n_failed=$((n_failed + 1))
            echo "!! training failed, see $run_dir/train.log" >&2
        fi
        (( DRY_RUN )) || printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$(date -Is)" "$d" "$RUN_TAG" "$seed" "$status" "$((SECONDS - start))" \
            "$run_dir/train.log" >> "$SUMMARY"
    done
done

echo
if (( n_failed )); then
    echo "Finished with $n_failed failure(s). See $SUMMARY"
    exit 1
fi
echo "Done. Summarize with: $PYTHON benchmark_summary.py --run_root $RUN_ROOT"
