#!/bin/bash
# Refresh the public repository github.com/oleksandr-barskyi/aimo-interp-report
# from the tracked files of solutions/, analysis/ and report/. Internal notes
# (report/README.md, report/OUTLINE.md) stay out; the built PDFs go in.
#   ./export-public.sh            copy into ../public and show git status there
set -e
cd "$(dirname "$0")/.."
DEST=public
mkdir -p "$DEST"
git ls-files solutions analysis report \
  | grep -v -E '^report/(README|OUTLINE)\.md$' \
  | while read -r f; do mkdir -p "$DEST/$(dirname "$f")"; cp -p "$f" "$DEST/$f"; done
for f in report/report.pdf report/report-main.pdf report/neurips_2026.sty \
         report/neurips-header.tex report/submissions.py report/submissions-output.txt \
         report/submissions.json report/export-public.sh; do
  [ -f "$f" ] && cp -p "$f" "$DEST/$f"
done
cp -p report/public-README.md "$DEST/README.md"
cp -p report/public-LICENSE "$DEST/LICENSE"
printf '__pycache__/\n*.pyc\nreport/mainonly.md\n' > "$DEST/.gitignore"
git -C "$DEST" status --short | head -40
