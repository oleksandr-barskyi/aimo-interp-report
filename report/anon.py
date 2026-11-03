"""Make the anonymous review copy of report.md for OpenReview.

The OpenReview form asks for a PDF without author names, so the author block,
the repository URL and the Codabench user name are replaced; the code itself
goes to OpenReview as supplementary_material.zip.
"""
import re
import sys

REPO = "`https://github.com/oleksandr-barskyi/aimo-interp-report`"
r = sys.stdin.read()
r = re.sub(r'^author: .*\n', '', r, count=1, flags=re.M)
r = r.replace("**Repository.** " + REPO, "**Repository.** The supplementary material (a public repository link is withheld for anonymous review)")
r = r.replace(REPO, "the supplementary material (a public repository link is withheld for anonymous review)")
r = r.replace("Codabench user `obarskyi`, ", "")
r = r.replace("`obarskyi`", "(withheld)")
for s in ("barskyi", "umka", "karazin"):
    if s in r.lower():
        sys.exit(f"anon.py: '{s}' still in the text")
sys.stdout.write(r)
