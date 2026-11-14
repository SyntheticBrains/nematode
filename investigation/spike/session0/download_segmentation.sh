#!/bin/bash
# Download DANDI 001075 segmentation NWB files listed in a TSV (asset_id, path, size)
# into a destination directory, flattening the path to its basename. 4 at a time.
# Usage: download_segmentation.sh <assets.tsv> <dest_dir>
set -u
tsv="$1"
dest="$2"
mkdir -p "$dest"
tail -n +2 "$tsv" | while IFS=$'\t' read -r aid path size; do
  echo "$aid $(basename "$path") $size"
done | xargs -P 4 -n 3 sh -c '
  aid="$0"; name="$1"; size="$2"; out="'"$dest"'/$name"
  if [ -f "$out" ] && [ "$(stat -f %z "$out")" = "$size" ]; then exit 0; fi
  curl -sS -L --retry 3 -o "$out" "https://api.dandiarchive.org/api/assets/$aid/download/"
  got=$(stat -f %z "$out")
  if [ "$got" != "$size" ]; then echo "SIZE MISMATCH $name $got != $size"; else echo "ok $name"; fi
'
echo DONE
