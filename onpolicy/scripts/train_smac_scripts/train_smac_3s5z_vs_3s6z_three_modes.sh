#!/bin/sh
set -eu
case "$0" in
    */*) script_path=$0 ;;
    *) script_path="./$0" ;;
esac
script_dir=$(CDPATH= cd "${script_path%/*}" && pwd)
set -- "3s5z_vs_3s6z" "$@"
. "$script_dir/train_smac_three_modes.sh"
