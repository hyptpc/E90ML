#!/usr/bin/env bash
# Usage: bash RunSaveRoot.sh CONFIG_FILE [CLEAN_BASE_DIR]
#   CONFIG_FILE    : path to config (required)
#   CLEAN_BASE_DIR : directory in which to remove build artifacts (default: repo root)
# Note: Run this script with bash (not via `root -l -q`), otherwise ROOT will try to parse it as a macro.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="${script_dir}"
macro="${repo_dir}/src/SaveRoot.C"
input_dir="${repo_dir}/data/G4rootfile"
output_dir="${repo_dir}/data/input"
clean_base="${repo_dir}"

if [[ $# -lt 1 ]]; then
  echo "ERROR: CONFIG_FILE is required. Usage: bash RunSaveRoot.sh CONFIG_FILE [CLEAN_BASE_DIR]" >&2
  exit 1
fi

config_file="$1"
if [[ $# -ge 2 ]]; then
  clean_base="$2"
fi

trim() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "$s"
}

command -v root >/dev/null 2>&1 || { echo "root command not found in PATH." >&2; exit 1; }
[[ -f "$macro" ]] || { echo "Macro not found: $macro" >&2; exit 1; }
[[ -f "$config_file" ]] || { echo "Config not found: $config_file" >&2; exit 1; }

mkdir -p "$output_dir"

# Default reaction file basenames
declare -A name_map=(
  [SigmaNCusp]="SigmaNCusp"
  [QFLambda]="QFLambda"
  [QFSigmaZ]="QFSigmaZ"
)

# Override by config (format: Key=Value, e.g. SigmaNCusp=MyFileName)
while IFS='=' read -r key value; do
  [[ -z "$key" || "$key" =~ ^[[:space:]]*# ]] && continue
  key="$(trim "${key%%#*}")"
  value="$(trim "${value%%#*}")"
  case "$key" in
    SigmaNCusp|QFLambda|QFSigmaZ)
      [[ -n "$value" ]] && name_map["$key"]="$value"
      ;;
    *)
      ;;
  esac
done < "$config_file"

names=(
  "${name_map[SigmaNCusp]}"
  "${name_map[QFLambda]}"
  "${name_map[QFSigmaZ]}"
)
labels=(1 2 3)

for idx in "${!names[@]}"; do
  name="${names[$idx]}"
  label="${labels[$idx]}"
  src_name="$name"
  [[ "$src_name" != *.root ]] && src_name="${src_name}.root"
  src="${input_dir}/${src_name}"
  dest="${output_dir}/${src_name}"

  [[ -f "$src" ]] || { echo "Input not found: $src" >&2; exit 1; }

  echo "Running SaveRoot for ${name} (label=${label})"
  root -l -b -q "${macro}(\"${src}\",\"${dest}\",${label})"
done

# remove file types
EXTENSIONS=(
  "d"
  "so"
  "cxx"
  "pcm"
)

TARGET_DIR="${clean_base}"

for ext in "${EXTENSIONS[@]}"; do
  find "$TARGET_DIR" -type f -name "*.$ext" -exec rm -v {} +
done

echo "Finished. Outputs are in ${output_dir}"
