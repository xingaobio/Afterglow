#!/usr/bin/env bash
# Count files by extension on a photo drive (case-insensitive), most frequent first.
# Usage: scripts/count_library.sh /media/xin/PHOTOS
set -euo pipefail

if [[ $# -ne 1 || ! -d "$1" ]]; then
  echo "usage: $0 <directory>   # count files per extension, e.g. $0 /media/xin/PHOTOS" >&2
  exit 1
fi

find "$1" -type f -not -path '*/.*' -not -path '*/@eaDir/*' -not -path '*/System Volume Information/*' \
  | sed -E 's/.*\.([^./]+)$/\1/;t;s/.*/(none)/' \
  | tr '[:upper:]' '[:lower:]' \
  | sort | uniq -c | sort -rn
