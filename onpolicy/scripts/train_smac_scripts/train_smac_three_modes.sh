#!/bin/sh

set -eu

usage() {
    cat <<'EOF'
Run GAE, historical SAE, and pure SAE sequentially on one SMAC map.

Usage:
  sh train_smac_three_modes.sh MAP [options]

Options:
  --seed N       Run one seed only.
  --seed-min N   Override the first seed.
  --seed-max N   Override the last seed.
  --gpu ID       CUDA_VISIBLE_DEVICES value (default: 0).
  --python PATH  Python executable (default: python).
  --dry-run      Print all commands without starting training.
  -h, --help     Show this help.

Examples:
  sh train_smac_three_modes.sh 3m --seed 1 --gpu 0
  sh train_smac_three_modes.sh corridor --seed-min 1 --seed-max 7
  sh train_smac_three_modes.sh MMM2 --seed 1 --dry-run

Optional environment overrides:
  EXPERIMENT_PREFIX, ALGORITHM_NAME, LR, CRITIC_LR, CLIP_PARAM,
  PPO_EPOCH, NUM_ENV_STEPS, EPISODE_LENGTH, N_ROLLOUT_THREADS,
  N_TRAINING_THREADS, EVAL_EPISODES, USER_NAME
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 2
}

is_positive_integer() {
    case "$1" in
        ''|*[!0-9]*) return 1 ;;
        *) [ "$1" -ge 1 ] ;;
    esac
}

if [ "$#" -eq 0 ]; then
    usage >&2
    exit 2
fi

case "$1" in
    -h|--help)
        usage
        exit 0
        ;;
esac

requested_map=$1
shift

single_seed=''
seed_min_override=''
seed_max_override=''
gpu=${CUDA_VISIBLE_DEVICES:-0}
python_bin=${PYTHON_BIN:-python}
dry_run=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        --seed)
            [ "$#" -ge 2 ] || die '--seed requires a value'
            single_seed=$2
            shift 2
            ;;
        --seed-min)
            [ "$#" -ge 2 ] || die '--seed-min requires a value'
            seed_min_override=$2
            shift 2
            ;;
        --seed-max)
            [ "$#" -ge 2 ] || die '--seed-max requires a value'
            seed_max_override=$2
            shift 2
            ;;
        --gpu)
            [ "$#" -ge 2 ] || die '--gpu requires a value'
            gpu=$2
            shift 2
            ;;
        --python)
            [ "$#" -ge 2 ] || die '--python requires a value'
            python_bin=$2
            shift 2
            ;;
        --dry-run)
            dry_run=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

[ -z "$single_seed" ] || is_positive_integer "$single_seed" || \
    die '--seed must be a positive integer'
[ -z "$seed_min_override" ] || is_positive_integer "$seed_min_override" || \
    die '--seed-min must be a positive integer'
[ -z "$seed_max_override" ] || is_positive_integer "$seed_max_override" || \
    die '--seed-max must be a positive integer'
[ -z "$single_seed" ] || \
    { [ -z "$seed_min_override" ] && [ -z "$seed_max_override" ]; } || \
    die '--seed cannot be combined with --seed-min or --seed-max'
[ -n "$gpu" ] || die '--gpu cannot be empty'
[ -n "$python_bin" ] || die '--python cannot be empty'

# Defaults mirror the existing SMAC scripts. Map-specific cases below change
# only the values that were explicitly different in those scripts.
map_name=$requested_map
default_algorithm_name='rmappo'
default_seed_min=1
default_seed_max=1
default_ppo_epoch=15
default_lr='5e-4'
default_critic_lr='5e-4'
default_clip_param='0.2'
default_num_mini_batch=1
default_gain='0.01'
use_stacked_frames=0
stacked_frames=1

case "$requested_map" in
    3m|8m|2m_vs_1z|2s_vs_1sc|1c3s5z|MMM)
        ;;
    10m_vs_11m)
        default_ppo_epoch=10
        ;;
    27m_vs_30m|3s5z|3s5z_vs_3s6z|2c_vs_64zg)
        default_ppo_epoch=5
        ;;
    5m_vs_6m)
        default_ppo_epoch=10
        default_clip_param='0.05'
        ;;
    8m_vs_9m)
        default_seed_min=6
        default_seed_max=10
        default_lr='1e-3'
        default_critic_lr='5e-4'
        default_clip_param='0.05'
        ;;
    25m)
        default_algorithm_name='mappo'
        default_seed_max=10
        default_lr='1e-3'
        default_critic_lr='5e-4'
        default_clip_param='0.05'
        ;;
    3s_vs_3z)
        default_algorithm_name='mappo'
        default_seed_min=11
        default_seed_max=20
        default_lr='1e-3'
        default_critic_lr='1e-3'
        default_clip_param='0.15'
        ;;
    3s_vs_4z)
        default_algorithm_name='mappo'
        default_seed_max=20
        default_lr='1e-3'
        default_critic_lr='5e-4'
        default_clip_param='0.1'
        ;;
    3s_vs_5z)
        default_algorithm_name='mappo'
        default_clip_param='0.05'
        use_stacked_frames=1
        stacked_frames=4
        ;;
    6h_vs_8z|corridor)
        default_algorithm_name='mappo'
        default_ppo_epoch=5
        ;;
    MMM2)
        default_ppo_epoch=5
        default_num_mini_batch=2
        default_gain='1'
        ;;
    baneling|so_many_baneling)
        map_name='so_many_baneling'
        default_algorithm_name='mappo'
        default_seed_max=10
        default_lr='1e-3'
        default_critic_lr='5e-4'
        default_clip_param='0.05'
        ;;
    bane_vs_bane)
        default_seed_min=11
        default_seed_max=20
        default_lr='1e-3'
        default_critic_lr='5e-4'
        default_clip_param='0.05'
        ;;
    *)
        die "unknown SMAC map '$requested_map'"
        ;;
esac

if [ -n "$single_seed" ]; then
    first_seed=$single_seed
    last_seed=$single_seed
else
    first_seed=${seed_min_override:-$default_seed_min}
    last_seed=${seed_max_override:-$default_seed_max}
fi

[ "$first_seed" -le "$last_seed" ] || \
    die '--seed-min must be less than or equal to --seed-max'

case "$0" in
    */*) script_path=$0 ;;
    *) script_path="./$0" ;;
esac
script_dir=$(CDPATH= cd "${script_path%/*}" && pwd)
repo_root=$(CDPATH= cd "$script_dir/../../.." && pwd)
train_script="$repo_root/onpolicy/scripts/train/train_smac.py"

if [ "$dry_run" -eq 0 ] && [ ! -f "$train_script" ]; then
    die "training entry point not found: $train_script"
fi

export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"

env_name='StarCraft2'
algorithm_name=${ALGORITHM_NAME:-$default_algorithm_name}
experiment_prefix=${EXPERIMENT_PREFIX:-smac_three_modes}
lr=${LR:-$default_lr}
critic_lr=${CRITIC_LR:-$default_critic_lr}
clip_param=${CLIP_PARAM:-$default_clip_param}
ppo_epoch=${PPO_EPOCH:-$default_ppo_epoch}
num_env_steps=${NUM_ENV_STEPS:-10000000}
episode_length=${EPISODE_LENGTH:-400}
n_rollout_threads=${N_ROLLOUT_THREADS:-8}
n_training_threads=${N_TRAINING_THREADS:-1}
eval_episodes=${EVAL_EPISODES:-32}
user_name=${USER_NAME:-jimzhao0422}

run_one() {
    mode=$1
    seed=$2
    experiment_name="${experiment_prefix}_${mode}"

    set -- "$python_bin" "$train_script" \
        --env_name "$env_name" \
        --algorithm_name "$algorithm_name" \
        --experiment_name "$experiment_name" \
        --map_name "$map_name" \
        --seed "$seed" \
        --n_training_threads "$n_training_threads" \
        --n_rollout_threads "$n_rollout_threads" \
        --num_mini_batch "$default_num_mini_batch" \
        --episode_length "$episode_length" \
        --num_env_steps "$num_env_steps" \
        --ppo_epoch "$ppo_epoch" \
        --gain "$default_gain" \
        --lr "$lr" \
        --critic_lr "$critic_lr" \
        --clip_param "$clip_param" \
        --use_value_active_masks \
        --use_eval \
        --eval_episodes "$eval_episodes"

    if [ "$use_stacked_frames" -eq 1 ]; then
        set -- "$@" --stacked_frames "$stacked_frames" --use_stacked_frames
    fi

    case "$mode" in
        gae)
            ;;
        sae_legacy)
            set -- "$@" \
                --use_gae \
                --sae_alpha 0.1 \
                --sae_omega 1.0 \
                --sae_gate hard \
                --sae_temperature 1.0 \
                --sae_blend_mode legacy_add
            ;;
        sae_pure)
            set -- "$@" \
                --use_gae \
                --use_pure_sae \
                --sae_alpha 0.1 \
                --sae_omega 1.0 \
                --sae_gate hard \
                --sae_temperature 1.0 \
                --sae_blend_mode legacy_add
            ;;
        *)
            die "internal error: unknown mode '$mode'"
            ;;
    esac

    # Existing SMAC scripts use shared policy, so --share_policy is
    # intentionally absent. --use_value_active_masks is store_false here and
    # is retained to match the original map scripts.
    set -- "$@" --user_name "$user_name"

    printf '\n[%s] map=%s seed=%s experiment=%s\n' \
        "$mode" "$map_name" "$seed" "$experiment_name"

    if [ "$dry_run" -eq 1 ]; then
        printf 'DRY_RUN CUDA_VISIBLE_DEVICES=%s' "$gpu"
        printf ' %s' "$@"
        printf '\n'
    elif CUDA_VISIBLE_DEVICES="$gpu" "$@"; then
        printf '[%s] seed=%s completed\n' "$mode" "$seed"
    else
        status=$?
        printf '[%s] seed=%s failed with status %s\n' \
            "$mode" "$seed" "$status" >&2
        exit "$status"
    fi
}

printf 'map=%s algorithm=%s modes=gae,sae_legacy,sae_pure seeds=%s..%s gpu=%s\n' \
    "$map_name" "$algorithm_name" "$first_seed" "$last_seed" "$gpu"
printf 'lr=%s critic_lr=%s clip=%s ppo_epoch=%s env_steps=%s\n' \
    "$lr" "$critic_lr" "$clip_param" "$ppo_epoch" "$num_env_steps"

for mode in gae sae_legacy sae_pure; do
    seed=$first_seed
    while [ "$seed" -le "$last_seed" ]; do
        run_one "$mode" "$seed"
        seed=$((seed + 1))
    done
done

if [ "$dry_run" -eq 1 ]; then
    printf '\nDry run completed; no training was started.\n'
else
    printf '\nAll three modes completed for %s.\n' "$map_name"
fi
