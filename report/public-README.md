# A leaderboard that cannot measure

Code, submission bundles and paper source for the technical report
*"A leaderboard that cannot measure: negative results and measurement limits
in the AIMO Interpretability Challenge"* (NeurIPS 2026 competition track,
AIMO Interpretability Challenge, Codabench competition 16180, user `obarskyi`).

The paper: [`report/report.pdf`](report/report.pdf) (full) and
[`report/report-main.pdf`](report/report-main.pdf) (main body only).

## Layout

| path | what |
| --- | --- |
| `solutions/` | every submitted bundle as a source directory; `self-consistency/` is the method of §5, `census*/` the composition probes of §6 and Appendix B |
| `analysis/` | the 80-pair analysis of §4-§5 (`aimo_reference.py`) and the corpus work of Appendix E |
| `report/` | the paper source (`report.md`), the scripts behind every number, their saved outputs and both figures |

## Reproducing

Python 3.11+, `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`.

```bash
python3 report/power.py       # §3 power numbers, Appendix C   -> report/power-output.txt
python3 report/runnoise.py    # §3 within-test model, App. D   -> report/runnoise-output.txt
python3 report/ci80.py        # §4 clustered intervals          -> report/ci80-output.txt
python3 report/figures.py     # Figures 1 and 2
```

The analysis scripts read the organizers' public data from the repository
root. Fetch it once:

```bash
python3 analysis/fetch_hf.py      # aimo-interp HF datasets -> data-hf/
git clone https://github.com/aimo-interp/getting-started   # val-sample, harness
git clone https://github.com/aimo-interp/baselines         # math-robust-final.csv
(cd getting-started && git checkout de79405 && python3 scripts/import_hf_dataset.py)
```

A submission bundle is the deterministic ZIP of one `solutions/<name>`
directory, built by `scripts/build.py` of
[`aimo-interp/getting-started`](https://github.com/aimo-interp/getting-started)
at commit `de79405`; Small-track bundles carry the organizers' `small.txt`
marker. `report/submissions.py` lists every submission with its scores
(needs a Codabench login in `CB_PASS`).

The PDF is typeset with the official NeurIPS 2026 style (`neurips_2026.sty`)
through pandoc and pdflatex: `report/build-pdf.sh` (full) and
`report/build-pdf.sh --main` (main body, the 5-page part).

## License

MIT, see `LICENSE`. The NeurIPS style file is redistributed as published by
NeurIPS.
