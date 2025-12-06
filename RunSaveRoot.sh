#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
macro="${script_dir}/SaveRoot.C"
input_dir="${script_dir}/data/G4rootfile"
output_dir="${script_dir}/data/input"

command -v root >/dev/null 2>&1 || { echo "root command not found in PATH." >&2; exit 1; }
[[ -f "$macro" ]] || { echo "Macro not found: $macro" >&2; exit 1; }

mkdir -p "$output_dir"

names=(SigmaNCusp QFLambda QFSigmaZ) # Change file names in your way
labels=(1 2 3)

for idx in "${!names[@]}"; do
  name="${names[$idx]}"
  label="${labels[$idx]}"
  src="${input_dir}/${name}.root"
  dest="${output_dir}/${name}.root"

  [[ -f "$src" ]] || { echo "Input not found: $src" >&2; exit 1; }

  echo "Running SaveRoot for ${name} (label=${label})"
  root -l -b -q "\"${macro}\"(\"${src}\",\"${dest}\",${label})"
done

echo "Finished. Outputs are in ${output_dir}"
