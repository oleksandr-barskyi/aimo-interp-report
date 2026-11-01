#!/bin/bash
# Typeset report.md with the official NeurIPS 2026 style (neurips_2026.sty from
# media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip,
# option sglblindworkshop) through pandoc + pdflatex.
#   ./build-pdf.sh          full report (body, limitations, AI use statement,
#                           references, appendices)
#   ./build-pdf.sh --main   main body only, the part under the 5-page limit
set -e
cd "$(dirname "$0")"
SRC=report.md
OUT=report.pdf
if [ "$1" = "--main" ]; then
  PYTHONIOENCODING=utf-8 python3 - <<'PY'
r = open('report.md').read()
open('mainonly.md', 'w').write(r[:r.index('# Limitations')])
PY
  SRC=mainonly.md
  OUT=report-main.pdf
  shift
fi
PYTHONIOENCODING=utf-8 python3 tex-safe.py < "$SRC" > "${TMPDIR:-/tmp}/report-tex-safe.md"
pandoc "${TMPDIR:-/tmp}/report-tex-safe.md" -o "$OUT" \
  --pdf-engine=pdflatex \
  -V documentclass=article -V fontfamily=times -V colorlinks=true \
  -H neurips-header.tex \
  --resource-path=. "$@"
pdfinfo "$OUT" | grep -i pages
