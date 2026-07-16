#!/bin/sh

set -eu

usage() {
    cat <<'EOF'
Run GAE, historical SAE, and pure SAE sequentially on one MPE scenario.

Usage:
  sh train_mpe_three_modes.sh {comm|reference|spread} [options]

Options:
  --seed N       Run one seed only.
  --seed-max N   Run seeds 1..N for every mode.
  --gpu ID       CUDA_VISIBLE_DEVICES value (default: 0).
  --python PATH  Python executable (default: python).
  --dry-run      Print all commands without starting training.
  -h, --help     Show this help.

Examples:
  sh train_mpe_three_modes.sh spread --seed 3 --gpu 0
  sh train_mpe_three_modes.sh reference --seed-max 7 --gpu 0
  sh train_mpe_three_modes.sh comm --seed 1 --dry-run

Optional environment overrides:
  EXPERIMENT_PREFIX, ALGORITHM_NAME, LR, CRITIC_LR, NUM_ENV_STEPS,
  N_ROLLOUT_THREADS, N_TRAINING_THREADS, USER_NAME
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

target=$1
shift

single_seed=''
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
[ -z "$seed_max_override" ] || is_positive_integer "$seed_max_override" || \
    die '--seed-max must be a positive integer'
[ -z "$single_seed" ] || [ -z "$seed_max_override" ] || \
    die '--seed and --seed-max cannot be used together'
[ -n "$gpu" ] || die '--gpu cannot be empty'
[ -n "$python_bin" ] || die '--python cannot be empty'

case "$target" in
    comm|speaker_listener|simple_speaker_listener)
        scenario='simple_speaker_listener'
        num_agents=2
        num_landmarks=3
        default_seed_max=3
        default_num_env_steps=2000000
        ppo_epoch=15
        default_lr='1.5e-3'
        spread_activation_flag=0
        ;;
    reference|simple_reference)
        scenario='simple_reference'
        num_agents=2
        num_landmarks=3
        default_seed_max=10
        default_num_env_steps=4000000
        ppo_epoch=15
        default_lr='7e-4'
        spread_activation_flag=0
        ;;
    spread|simple_spread)
        scenario='simple_spread'
        num_agents=3
        num_landmarks=3
        default_seed_max=5
        default_num_env_steps=8000000
        ppo_epoch=10
        default_lr='7e-4'
        spread_activation_flag=1
        ;;
    *)
        die "unknown environment '$target'; use comm, reference, or spread"
        ;;
esac

if [ -n "$single_seed" ]; then
    first_seed=$single_seed
    last_seed=$single_seed
elif [ -n "$seed_max_override" ]; then
    first_seed=1
    last_seed=$seed_max_override
else
    first_seed=1
    last_seed=$default_seed_max
fi

case "$0" in
    */*) script_path=$0 ;;
    *) script_path="./$0" ;;
esac
script_dir=$(CDPATH= cd "${script_path%/*}" && pwd)
repo_root=$(CDPATH= cd "$script_dir/../../.." && pwd)
train_script="$repo_root/onpolicy/scripts/train/train_mpe.py"

if [ "$dry_run" -eq 0 ] && [ ! -f "$train_script" ]; then
    die "training entry point not found: $train_script"
fi

export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"

env_name='MPE'
algorithm_name=${ALGORITHM_NAME:-mappo}
experiment_prefix=${EXPERIMENT_PREFIX:-three_modes}
lr=${LR:-$default_lr}
critic_lr=${CRITIC_LR:-$lr}
num_env_steps=${NUM_ENV_STEPS:-$default_num_env_steps}
n_rollout_threads=${N_ROLLOUT_THREADS:-128}
n_training_threads=${N_TRAINING_THREADS:-1}
user_name=${USER_NAME:-jimzhao0422}

run_one() {
    mode=$1
    seed=$2
    experiment_name="${experiment_prefix}_${mode}"

    set -- "$python_bin" "$train_script" \
        --env_name "$env_name" \
        --algorithm_name "$algorithm_name" \
        --experiment_name "$experiment_name" \
        --scenario_name "$scenario" \
        --num_agents "$num_agents" \
        --num_landmarks "$num_landmarks" \
        --seed "$seed" \
        --n_training_threads "$n_training_threads" \
        --n_rollout_threads "$n_rollout_threads" \
        --num_mini_batch 1 \
        --episode_length 25 \
        --num_env_steps "$num_env_steps" \
        --ppo_epoch "$ppo_epoch" \
        --gain 0.01 \
        --lr "$lr" \
        --critic_lr "$critic_lr"

    # Preserve the existing spread experiment's activation flag. In this
    # repository --use_ReLU is a store_false flag.
    if [ "$spread_activation_flag" -eq 1 ]; then
        set -- "$@" --use_ReLU
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

    # --share_policy is store_false here; retaining it keeps all three runs on
    # the same separated-policy path.
    set -- "$@" --user_name "$user_name" --share_policy

    printf '\n[%s] scenario=%s seed=%s experiment=%s\n' \
        "$mode" "$scenario" "$seed" "$experiment_name"

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

printf 'scenario=%s modes=gae,sae_legacy,sae_pure seeds=%s..%s gpu=%s\n' \
    "$scenario" "$first_seed" "$last_seed" "$gpu"

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
    printf '\nAll three modes completed for %s.\n' "$scenario"
fi
