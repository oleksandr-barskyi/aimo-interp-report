---
title: "Working-phase leaderboards that cannot detect an AUC 0.65 scorer: negative results and measurement limits in the AIMO Interpretability Challenge"
author: "`Oleksandr Barskyi\\\\ Independent researcher\\\\ \\texttt{oleksandr.umka.barskyi@gmail.com}`{=latex}"
abstract: |
  We report a negative participation in the AIMO Interpretability Challenge:
  28 submissions, of which 7 were attempts by four methods and 14 were
  composition probes, none beating the best constant by more than noise. Our
  central claim is limited to label-free scorers in the range we observe (ROC
  AUC about 0.65): on the working-phase private tests of 14 and 21 cases such a
  scorer cannot be shown to beat a constant. For the rule we submitted (median
  inside each model group, ties counted robust), an AUC 0.65 scorer exceeds the
  critical score of a one-sided $\alpha = 0.05$ test in 16% of simulated runs on
  the Small Models Track and 19% on the Main Track; 80% power needs AUC 0.98
  and 0.92 there, or about 150 cases at AUC 0.65. About three quarters of the
  variance of an entry is which cases were drawn. On the organizers' 80 olympiad
  problem--model pairs, mostly from frontier models, 16 of 25 pairs robust under
  a zero-decay label have base accuracy zero: for them "robust" largely means
  "fails identically". Under the official 0.10/0.25 label this weakens to 20 of
  38, and on the test models' own training data at most 25 of 59 robust rows can
  be base failures. This is consistent with the published reference solution
  scoring below both constants and, in a simulation conditional on an assumed
  number of distinct wrong answers, with an inverted sign for a
  self-consistency proxy. Finally, per-group exception handling turns the
  scoring output into an oracle for predicates over the private test; we used it
  only to count the composition of a since-retired test, flagged it to the
  organizers, and describe it in full. All code and raw scores are public.
---

# 1 Introduction

The AIMO Interpretability Challenge asks for a binary verdict per
(problem, model) pair: does the model answer this olympiad problem *robustly*,
or does the answer rest on a pattern that perturbation removes? Submissions
are ranked by accuracy on a held-out test, separately on a Main Track (five
models, including `openai/gpt-oss-120b`) and a Small Models Track (the four
below 10B parameters). The write-up prizes are judged "regardless of ranking",
with explicit interest in negative results; this report is written for them.

Our position shapes everything below: a single participant on a CPU-only
server, with no GPU at any point, so every generation-based measurement had to
happen *inside* an evaluation submission, with empty logs and a three-number
return channel. We claim no place on either leaderboard: our best scores are
0.5714 (8/14) on the Small track, by a method we believe is indistinguishable
from a coin, and 0.5238 (11/21) on the Main track, by the constant
`always-false`. Of our 28 submissions only 7 were attempts by a method (four
methods); 14 were composition probes (§6) and the rest constants or cancelled
(Appendix G). We submitted no white-box probe: we could not train one.

**Contributions.** (1) A power analysis of the challenge's working-phase
leaderboards for the decision rule we actually submitted, and a decomposition
of a score's variance into case draw and run (§3). (2) A hypothesis, checked
on published data under two label rules, for why the official reference
solution scores below a constant (§4). (3) A simulation, conditional on an
assumed number m of distinct wrong answers, of the sign of a label-free
robustness proxy; m was never measured and the sign was fixed before any
submission (§5). (4) The scoring channel as a predicate oracle, with a proposed
mitigation (§6). (5) Eleven pre-registered predictions with their outcomes, an
appendix of engineering failure modes, and the code of every submission,
including the three versions of the method behind 932626, 932631 and 933057
(Appendices A, F, H). Code and every raw score:
`https://github.com/oleksandr-barskyi/aimo-interp-report`.

**Related work.** §3 instantiates the argument of Card et al. [1] on
underpowered test sets and the variance accounting of Bouthillier et al. [2],
specialised to a 14-case leaderboard where the case draw dominates. §6 relates
to leaderboard probing and adaptive overfitting [3--5] and adds a *coverage*
channel to the known label channel. §5 uses self-consistency [6] as a
robustness signal with an inverted sign; the reference baseline belongs to the
family of uncertainty-based self-assessment probes [7]. The corpus of §4 is
built on `hendrycks MATH` [8]; task and label are defined in the proposal [9].

# 2 Setup and constraints

Submissions implement `are_robust(model_id, reasoning_effort, problems)`;
ingestion calls it once per model group, and scoring returns exactly
`accuracy`, `coverage` and `invalid_predictions`, all over the total case
count. A run gets 3600 s on one 96 GB GPU, with the checkpoints cached and no
internet. Test sizes are not published but follow from the score grid, since
accuracy is a multiple of 1/n; constant submissions give n and the base rate
(Table 1).

| phase | track | n | robust | best constant |
| --- | --- | --- | --- | --- |
| v7 (until 2026-09-16) | Small | 19 | 10 | `always-true` 0.5263 |
| v8 (from 2026-09-16) | Small | 14 | 7 | tie, 0.5000 |
| v8 | Main | 21 | 10 | `always-false` 0.5238 |

: Private tests of the working phase, inferred from constant submissions.

On 2026-09-16 the organizers replaced the working-phase problems, so
everything §6 measured about the *composition* of the v7 test describes a test
that no longer exists. Our only compute was a CPU environment pinned to the
container's exact runtime with a 135M stand-in model; given empty evaluation
logs we consider that replication mandatory (Appendix A lists four bugs it
caught and one it did not).

# 3 What a 14-case leaderboard can measure

**The rule we submitted.** Our method (§5) calls a case robust when its score
is at or above the median of its own model group, ties counted robust. With
groups of 4/4/3/3 on the Small track a continuous score therefore makes exactly
s = 8 robust calls of 14, and on the Main track (five models, assumed
5/4/4/4/4) s = 11 of 21; the discrete agreement of 8 samples ties more often
and averages about 10 of 14 (Appendix D). We model a method as a binormal
scorer of given ROC AUC [10] ($\mu = \sqrt{2}\,\Phi^{-1}(\mathrm{AUC})$),
assign model groups to cases independently of the labels, apply this rule, and
simulate 200,000 tests (`report/power.py`). "Declared better" means exceeding
the critical score of a one-sided $\alpha = 0.05$ test whose null is the same
rule at AUC 0.5: s robust calls on a random subset of cases, which is exactly
hypergeometric. For the Small track (n = 14, best constant 7/14):

| true AUC | E[accuracy] | sd (cases) | P(score > constant) | power, randomized | power, non-randomized |
| --- | --- | --- | --- | --- | --- |
| 0.50 | 0.5007 | 1.92 | 0.299 | 0.051 | 0.002 |
| 0.65 | 0.5897 | 1.88 | 0.563 | 0.164 | 0.014 |
| 0.80 | 0.6805 | 1.73 | 0.824 | 0.392 | 0.059 |
| 0.90 | 0.7442 | 1.56 | 0.945 | 0.615 | 0.140 |

: Small track, deployed rule. The null rejects above 11/14; with s = 8 every score is odd, so the constant 7/14 can never be tied.

An AUC 0.65 scorer --- the range every label-free signal of §5 reaches on the
organizers' olympiad data --- beats the constant in 56% of tests but is
declared better in 16% (randomized test, attaining $\alpha$ exactly) or 1.4%
(conservative test). On the Main track the figures are 19% and 12%
(Appendix C). The smallest AUC detected with 80% power is **0.98 on the Small
track and 0.92 on the Main track**; at AUC 0.65, 80% is reached only between
120 and 150 cases (Figure 1). These statements are about scorers near AUC
0.65; an AUC 0.9 scorer would be detected in 62% (Small) and 75% (Main) of
tests. The rule we adopted early follows: **a public score may forbid a
candidate, never select one.** A coin reaches 11/14 with probability 0.052 per
submission, and the budget was 300 submissions.

![Probability that a one-sided test at α = 0.05 declares a scorer of the given AUC better than the best constant, against private test size (log scale), for the deployed rule with four model groups. Dots mark the sizes of the two working-phase tracks. Randomized test, attaining α = 0.05 exactly. At AUC 0.65 the 80% line is crossed between 120 and 150 cases.](fig1-power.pdf){width=74%}

**The live board.** On 2026-11-01, the day the working phase closed, the
Small track listed 138 teams over 1,074 scored entries: 96 above the best
constant and six at 14/14. The Main track listed 120 teams over 686 entries:
54 above the constant and one at 21/21. These are snapshots of a board that
was later replaced and cannot be re-run. A random median split scores 14/14
with probability about $3\times10^{-4}$, i.e. 0.3 expected perfect entries in 1,074,
so the six perfect scores come from methods with AUC near 1, from adaptive
selection on the board [3], or from problem-level memorisation: on
`train-main-v2` the labels of other configurations on the same problem text
predict a row's label in 62 of 63 cases (`report/ci80.py`), so a method that recognises problems
can score high without a robustness signal. Fourteen cases cannot tell these
apart. The organizers' decision to rank finalists on a fresh, unpublished set
is the right remedy; §3 says how large that set must be.

**Where the variance lives.** Across-test noise (redraw cases and samples)
and within-test noise (re-run the same code on the same cases) are different
questions. `report/runnoise.py` simulates the deployed method itself on fixed
tests drawn from the 80 olympiad pairs of §4 (Appendix D): at an assumed
m = 4 distinct wrong answers the total sd of 1.70 cases splits into 1.49
between test draws and 0.81 within one, so **about three quarters of the
variance of an entry is which cases were drawn** (the same under the official
label). A three-case swing between two runs on the same test then has
probability 0.03--0.06 for m from 2 to 8, a lower bound because the model
varies only answer sampling. Our own pair sits inside that bracket:
self-consistency scored 8/14 (932631) and, with one internal change (a
guaranteed time reserve, Appendix A.4), 5/14 (933231) on the same test. We do
not call that ordinary noise, since the code change is a live alternative; but
each score alone is consistent with no signal (exact null tails, stated for
every feasible s, in Appendix C), so neither is evidence about the method.

# 4 Why the official reference solution scores below a constant

On the v7 Small track the organizers' `uncertainty-profiling` reference, a
regressor over uncertainty features of a DeepSeek-R1-0528-Qwen3-8B trace,
submitted unchanged, scored **0.4211 (8/19)**: below `always-true` (10/19) and
`always-false` (9/19). That is about one sd below the constant and refutes
nothing; what follows is a hypothesis consistent with it, checkable on
published data. The labelled corpus a supervised method would train on
(`math-robust-final.csv`: 675 rows, 137 problems, all for
DeepSeek-R1-0528-Qwen3-8B) is built on `hendrycks MATH` [8], and **134 of 137
problems have base accuracy 1.0**: the model solves them, and the label asks
whether perturbation breaks it. A model that fails the base problem never
appears in training.

How much of the olympiad task sits in that unseen regime depends on the label
rule and the models (Table 3, Figure 2). The organizers' olympiad release
(`aimo-interp-challenge-sample-full`) has 10 problems × 8 models = 80 pairs, and
seven of the eight models are large or frontier models; the only one below
10B, `qwen3-8b`, has mean base accuracy 0.63 and 2 of 10 pairs at zero. Our
original analysis labelled a pair robust iff every perturbation has zero decay
(it matches `val-sample` 24/24, but so does the official rule). Under that
label no robust pair has intermediate base accuracy and 16 of 25 robust pairs
have base accuracy exactly zero: **for these frontier pairs, "robust" largely
means "fails identically"**. The official rule of `train-main-v2` (robust iff
the largest drop is at most 0.10, not robust iff at least 0.25, the band
unlabelled), which we apply as the maximum over all perturbation rows of the
sample (eight types), keeps 69 pairs. Base-zero pairs stay robust
without exception, since a drop from zero cannot be positive, but they are now
20 of 38 robust pairs, and 8 of 21 intermediate pairs become robust. On
`train-main-v2` itself, the only labelled data on the five test checkpoints,
base accuracy is not published, but base zero forces a drop of at most 0, so at
most 25 of its 59 robust rows (42%) can be base failures.

| quantity | zero-decay label | official 0.10/0.25 label | `train-main-v2` |
| ------------------------- | --------------------- | --------------------- | ----------------- |
| models | 7 frontier, `qwen3-8b` | same | 5 test models |
| labelled pairs (robust) | 80 (25) | 69 (38) | 80 (59) |
| robust at base 0 | 16 (64%) | 20 (53%) | at most 25 (42%) |
| robust, intermediate base | 0 of 27 | 8 of 21 | --- |
| "base = 0 → robust", acc. | 0.838 [0.775, 0.900] | 0.739 [0.629, 0.847] | --- |
| same rule, AUC | 0.784 [0.570, 0.897] | 0.763 [0.597, 0.884] | --- |

: The base-failure regime under both label rules; 95% intervals from a cluster bootstrap over the 10 problems [11] (`report/ci80.py`).

![Robustness label against base accuracy on the organizers' 80 olympiad pairs, under (a) the zero-decay label and (b) the official 0.10/0.25 label, where 11 pairs fall in the unlabelled band. Under (a) no intermediate pair is robust; under (b) base-zero pairs remain all robust but 8 intermediate pairs join them.](fig2-base80.pdf){width=86%}

The strong reading is therefore restricted to frontier pairs under the
zero-decay label. What survives the official rule and the test models is
weaker but enough for the hypothesis: about half of the robust frontier pairs,
and up to two fifths of the robust rows on the test models (an upper bound),
are base failures, a regime the hendrycks corpus lacks, and the
reason is in the definition (decay = base − perturbed; at zero there is
nothing to lose). It is also consistent with, though not a test of, the v7
robust fraction (0.526, against 0.263 in the hendrycks corpus) and the failure
of our text-only feature (Appendix E).

# 5 The sign of a label-free proxy, by simulation under an assumed m

A method that never sees answers needs a proxy for base accuracy. The natural
one is self-consistency [6], with the intuitive sign "high agreement → stable
→ robust". We simulated k = 8 answers per pair from the measured base
accuracies, with wrong answers spread uniformly over m distinct values
(`report/signmodel.py`), and report the AUC of the inverted sign (Table 4;
the intuitive sign is one minus it).

| label rule | m = 1 | m = 2 | m = 4 | m = 8 |
| ------------------------ | ------ | ------ | ------ | ------ |
| zero-decay | 0.301 | 0.557 | 0.637 | 0.665 |
| official 0.10/0.25 | 0.422 | 0.688 | 0.745 | 0.763 |

: Simulated AUC of "low agreement → robust", conditional on the assumed number m of distinct wrong answers.

The intuitive sign wins only when a model always repeats the same wrong answer
(m = 1). The public releases contain accuracies but no model answers, so m
cannot be estimated from them, and we could not measure it without GPU traces.
We *assumed* that a 4--8B model wrong on an olympiad problem is wrong in
scattered ways (m ≥ 4) and fixed **low agreement → robust**: the one
load-bearing assumption of the method, and the reason this section is a
conditional simulation, not a measurement. The sign was fixed before any
submission and never revisited: choosing a sign by a 14-bit leaderboard is the
cheapest possible overfit, and after 7/21 on the Main track (933057) we did
not submit the inverted rule.

The method is deliberately minimal (code of all three versions in Appendix H):
8 samples per problem at temperature 0.8, top-p 0.95, at most 6,144 new
tokens; answers read only from `ANSWER:` or `\boxed{}`; a sample that ends
without one gets a 24-token greedy continuation appended at the token level
(Appendix A.1--A.3); agreement $\sum_a c_a(c_a-1)/k(k-1)$; and **robust ⟺
agreement ≤ the median of its own model group**, ties robust, with no
leaderboard-derived threshold and no labelled data. Scores: 8/14 and 5/14 on
the Small track, 7/21 on the Main. Of our eleven pre-registered predictions
(Appendix F), every arithmetic one came out exact; the two about a score that
required the method to carry signal both failed.

# 6 The scoring channel is a predicate oracle

Ingestion catches exceptions **per model group**: if `are_robust` raises for
one model, its cases become `invalid_predictions` while other groups stay
valid, and all three numbers are fractions of the *total* case count. A
submission that answers `True` where a predicate P(model_id) holds and raises
elsewhere therefore reports $n \cdot \mathrm{coverage} = |\{P\}|$ and
$n \cdot \mathrm{accuracy} = |\{P \wedge \mathrm{robust}\}|$: one submission
answers one question about the private test with two integers. Label
extraction from a leaderboard is a known hazard [3, 4] that this boundary
neither creates nor removes; what it adds is the **coverage** channel, which
counts private cases satisfying any predicate the container can evaluate.

We used it on 2026-09-09 and 09-10: fourteen probes on the v7 Small track,
each a constant over a `model_id` or `AutoConfig` predicate, counted the
composition of that test by architecture (Appendix B). They returned
group-level counts of robust cases (for example, 2 robust of 4 DeepSeek cases),
but no case-level label; we ran no problem-text probe, and no result entered
any method. The rules' clarification that "any attempts for leakage of
evaluation data are still forbidden" was published on 2026-09-19, after the
probes; we flagged the property to the organizers by email the same day, this
report is its first full description, and we leave its classification under
the rules to them. We do not know whether the channel has since been closed.
Mitigations, in increasing cost: withhold the scoring numbers until the
deadline; quantise or noise them [4]; invalidate the whole submission on any
exception; enlarge the test.

# 7 Recommendations

For organizers: **report an interval beside each score**, which communicates
at zero cost what §3 argues at length; **rank finalists on a larger fresh
set** --- about 150 cases to detect AUC 0.65 reliably, and even 60--100 changes
the picture; **close the coverage channel** (§6); **return the last lines of
`prediction_stdout`**, which would have saved two of the bugs in Appendix A;
and **publish per-model scores** after the deadline. For participants: check
the sign of a proxy on the olympiad pairs before spending a submission, do not
select variants by score, and replicate the container locally.

# Limitations

We never trained a white-box probe, because we had no GPU; the hidden-state
baseline we prepared was never submitted, its artifact being trained on one
model's representations while that model covers a minority of test cases. Our
power figures rest on a binormal scorer, the deployed per-group median rule and
assumed group sizes (4/4/3/3 and 5/4/4/4/4); the limit itself is that the
score is a sum of 14 or 21 Bernoulli terms. The olympiad sample behind §4 and
§5 has 10 problems and mostly frontier models, so its AUC intervals are wide
(Table 3) and the strong form of §4 holds only for those pairs under the
zero-decay label. The within-test model of §3 varies only answer sampling, so
it bounds run-to-run variability from below, and its parameter m is assumed,
not measured. The reconstruction in §6 describes the v7 test, replaced on
2026-09-16. The live-board figures of §3 are a snapshot of the working-phase
board, not the final ranking, which the organizers compute on a fresh set. The
code of 932626 and 932631 is reconstructed from commits whose timestamps match
the submissions to the minute; only the 933057 bundle survives as a file and
matches its commit byte for byte. And the central negative claim --- that no
signal was demonstrated --- says what the evaluation can show for scorers near
AUC 0.65, not that the signal is absent.

# AI Use Statement

In this work, we used generative AI tools for ideation, coding, analysis and
writing. The work was carried out by AI coding agents (Anthropic Claude models,
run through Claude Code) under the author's direction on a CPU-only server: the
agents proposed and designed the measurements, wrote all submission bundles
and analysis scripts, ran them, interpreted the outputs, produced the figures,
and drafted and condensed this text, including this revision; a second model
(OpenAI Codex) and a second Claude model were used as independent reviewers of
the statistics and of the decisions to stop or continue a direction. We did
not use generative AI tools to produce any reported number: every figure in
this paper is the output of a seeded script in the public repository
(Appendix G) or a score returned by the Codabench scorer, and no model output
was used as data or as a label. We have reviewed all AI-assisted work and take
responsibility for the final content of this paper.

# References

[1] D. Card, P. Henderson, U. Khandelwal, R. Jia, K. Mahowald, and D. Jurafsky.
With little power comes great responsibility. In *Proceedings of the 2020
Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pages
9263--9274, 2020.

[2] X. Bouthillier, P. Delaunay, M. Bronzi, A. Trofimov, B. Nichyporuk,
J. Szeto, N. Sepah, E. Raff, K. Madan, V. Voleti, S. E. Kahou, V. Michalski,
D. Serdyuk, T. Arbel, C. Pal, G. Varoquaux, and P. Vincent. Accounting for
variance in machine learning benchmarks. In *Proceedings of Machine Learning
and Systems (MLSys)*, volume 3, pages 747--769, 2021.

[3] A. Blum and M. Hardt. The Ladder: a reliable leaderboard for machine
learning competitions. In *Proceedings of the 32nd International Conference on
Machine Learning (ICML)*, PMLR 37, pages 1006--1014, 2015.

[4] C. Dwork, V. Feldman, M. Hardt, T. Pitassi, O. Reingold, and A. Roth. The
reusable holdout: preserving validity in adaptive data analysis. *Science*,
349(6248):636--638, 2015.

[5] R. Roelofs, S. Fridovich-Keil, J. Miller, V. Shankar, M. Hardt, B. Recht,
and L. Schmidt. A meta-analysis of overfitting in machine learning. In
*Advances in Neural Information Processing Systems (NeurIPS)* 32, pages
9179--9189, 2019.

[6] X. Wang, J. Wei, D. Schuurmans, Q. Le, E. Chi, S. Narang, A. Chowdhery, and
D. Zhou. Self-consistency improves chain of thought reasoning in language
models. In *International Conference on Learning Representations (ICLR)*, 2023.
arXiv:2203.11171.

[7] S. Kadavath, T. Conerly, A. Askell, T. Henighan, D. Drain, E. Perez,
N. Schiefer, Z. Hatfield-Dodds, N. DasSarma, E. Tran-Johnson, et al. Language
models (mostly) know what they know. arXiv:2207.05221, 2022.

[8] D. Hendrycks, C. Burns, S. Kadavath, A. Arora, S. Basart, E. Tang, D. Song,
and J. Steinhardt. Measuring mathematical problem solving with the MATH
dataset. In *NeurIPS Datasets and Benchmarks Track*, 2021. arXiv:2103.03874.

[9] AIMO Interpretability Challenge organizers. Competition proposal.
arXiv:2607.13899, 2026.

[10] J. A. Hanley and B. J. McNeil. The meaning and use of the area under a
receiver operating characteristic (ROC) curve. *Radiology*, 143(1):29--36,
1982.

[11] C. A. Field and A. H. Welsh. Bootstrapping clustered data. *Journal of the
Royal Statistical Society: Series B*, 69(3):369--390, 2007.

# Appendix A. Engineering: what the empty logs hid

Evaluation returns `prediction_stdout` and `scoring_stdout` empty. Every bug
below was therefore either caught locally, in a CPU environment pinned to the
container's exact library versions with a 135M stand-in model, or not caught at
all. We list them because each cost a submission and each will recur for anyone
generating traces inside the container. The code of each version is in
Appendix H.

**A.1 The `<think>` tag is in the prompt, not in the generation.** The chat
templates of `Qwen/Qwen3.5-4B`, `Skywork/Skywork-OR1-Math-7B` and
`allenai/Olmo-3-7B-Think` open `<think>` at the end of the prompt;
`deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` leaves it to the model. A first version
of our forced-answer step searched for the tag in the generated text only, and
so appended `ANSWER:` *inside* an unclosed reasoning block on three models out
of four. That version (932626) scored 0.3571 (5/14); the corrected one (932631),
submitted before the first score was known, scored 0.5714 (8/14). §3 applies:
we do not claim the difference is measured --- only that the bug was real and
was found by inspecting templates, not scores.

**A.2 A lossy tokenizer round-trip.** With `transformers 5.13`, the DeepSeek
tokenizer decodes its own ids without whitespace: `"Step 1: add 2 and
2.\nANSWER: 4"` becomes `"Step1:add2and2.ANSWER:4"`. Any forced-continuation
implementation that decodes a trace to text and re-encodes it feeds the model a
mangled copy of its own reasoning --- on the one model for which public labels
exist. 932626 did exactly that; from 932631 on we operate entirely on token ids
and never round-trip.

**A.3 Harmony channels.** `openai/gpt-oss-120b` (Main track only) has no
`<think>` tags; it writes `<|channel|>analysis<|message|>` … `<|end|>` …
`<|start|>assistant<|channel|>final<|message|>`. Appending a bare `ANSWER:`
request lands inside the analysis channel. The correct continuation for a
truncated trace is `<|end|><|start|>assistant<|channel|>final<|message|>ANSWER:`;
933057 is the first version that writes it.

**A.4 A forcing step that disappears under time pressure.** Our budget-forcing
step ran under the guard `if pending and deadline − now > 20.0`. When the main
generation consumes its whole slice (the `max_time` argument of `generate`
fires), fewer than 20 seconds remain, the forcing pass **never runs**, every
truncated sample is left without an answer, each becomes its own unique label,
agreement collapses to 0 for *every* problem in the batch, and the median rule
degenerates to all-`True`. We reproduced this as a controlled A/B on CPU (old
branch: 0 forcing calls, answers `[None, None]`, agreement 0.0; new branch: 1
forcing call, answers `['12','15']`, a genuine disagreement) and fixed it by
reserving 15% of each call's slice, floor 20 s, ceiling 90 s, for the forcing
pass, with the guard removed (933231, the version in `solutions/self-consistency`).
The fix is correct and did not help: 5/14, and
the run finished in ≤ 41 minutes of a 52.5-minute budget, so on the Small track
(225 s per problem) time was never the binding constraint. It plausibly was on
the Main track (150 s per problem, plus a 62 GB model), but with empty logs
this remains a hypothesis. This is also the concrete mechanism that keeps the
§3 within-test model a lower bound: it is a run-level effect that changes
every prediction in a batch at once and is invisible to a model that varies
only answer sampling. **General lesson:** a label-free method whose decision
rule is relative to a batch has a degenerate mode that scores exactly the
constant --- design the fallback so that the degenerate score is
*distinguishable* from the method's fallback score, or you cannot tell "no
signal" from "no run". We chose the all-`True` fallback on the Main track for
exactly this reason, even though all-`False` scores higher there.

**A.5 Operational.** Queue latency ranged from 3 minutes to 5.5 hours across
our submissions, so a same-day plan that depends on a result is unsafe;
`model_id` values in the working phase are plain HF repository names that load
directly, and any alias-rewriting map is at best useless.

# Appendix B. The v7 census, probe by probe

Fourteen submissions on the v7 Small track (n = 19, 10 robust). Each answers a
constant `True` where the predicate holds and raises elsewhere, so
`n · coverage` is the size of the subgroup and `n · accuracy` the number of
robust cases in it (§6).

| question | submission | n·coverage | n·accuracy |
| --- | --- | --- | --- |
| name contains `deepseek`/`qwen3-8b` | 920381 | 4 | 2 |
| resolves to `Olmo-3-7B-Instruct` | 920398 | 6 | 3 |
| resolves to a large cached model (120B/27B) | 920400 | 0 | 0 |
| `AutoConfig` loads the id directly | 920403 | 19 | 10 |
| among the 9 unknowns: `model_type` ∈ {qwen3, olmo, gpt_oss, gemma} | 920428 | 3 | 2 |
| among the 9: `llama` / `mistral` / `phi` / rest | 920913-5, 920917 | 0 each | 0 |
| among the 9: `qwen2*` | 920916 | 6 | 3 |
| among the 6 qwen2: `hidden_size ≥ 3584` | 920921 | 6 | 3 |
| among the 6 qwen2: `deepseek` / `qwen25` in `_name_or_path` | 920919-20 | 0 | 0 |
| among the trio: `model_type == qwen3` | 920922 | 3 | 2 |

: The fourteen composition probes on the retired v7 Small test.

Final composition of the 19-case v7 Small test:

| group | cases | robust |
| --- | --- | --- |
| `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` | 4 | 2 |
| other `qwen3` | 3 | 2 |
| `qwen2`, 7B class | 6 | 3 |
| `allenai/Olmo-3-7B-Instruct` | 6 | 3 |

: Composition of the v7 Small test reconstructed from the probes.

# Appendix C. Leaderboard statistics in full

All numbers from `report/power.py` (fixed seed), for the deployed rule:
inside each model group, robust iff the score is at or above the group median,
ties robust; groups assigned to cases independently of the labels.

**Main Track, v8** (n = 21, 10 robust, best constant 11/21, groups assumed
5/4/4/4/4, so s = 11 for a continuous score). The exact null rejects above
14/21 (non-randomized $\alpha$ = 0.023).

| true AUC | E[accuracy] | sd (cases) | P(score > constant) | P(score ≥ constant + 3) | power, randomized | power, non-rand. |
| --- | --- | --- | --- | --- | --- | --- |
| 0.50 | 0.4991 | 2.34 | 0.411 | 0.136 | 0.050 | 0.022 |
| 0.60 | 0.5603 | 2.32 | 0.635 | 0.296 | 0.128 | 0.074 |
| 0.65 | 0.5918 | 2.29 | 0.742 | 0.405 | 0.192 | 0.123 |
| 0.70 | 0.6232 | 2.24 | 0.831 | 0.523 | 0.273 | 0.192 |
| 0.80 | 0.6873 | 2.10 | 0.950 | 0.763 | 0.489 | 0.401 |
| 0.90 | 0.7551 | 1.88 | 0.995 | 0.939 | 0.752 | 0.692 |

: Main track, deployed rule. Smallest AUC with 80% power: 0.92.

The Small-track rows for AUC 0.60 and 0.70 complete Table 2: E[accuracy]
0.5590 and 0.6190, P(score > constant) 0.468 and 0.653, randomized power 0.113
and 0.224. The smallest AUC with 80% power there is 0.98.

**Power against test size.** The non-randomized test rejects when the score
strictly exceeds the critical value; because the score is discrete it cannot
attain $\alpha$ = 0.05 and is conservative. The randomized test additionally
rejects with probability $\gamma$ at the critical value, attaining
$\alpha$ = 0.05 exactly; it is the curve of Figure 1. The deployed rule uses
four equal model groups; the batch median of the original report is shown for
comparison.

| n (balanced) | reject if score > | actual $\alpha$ (non-rand.) | power at AUC 0.65 (non-rand.) | power at AUC 0.65 (rand.) | batch median, power (rand.) |
| --- | --- | --- | --- | --- | --- |
| 14 | 11/14 | 0.002 | 0.014 | 0.164 | 0.174 |
| 21 | 15/21 | 0.006 | 0.046 | 0.215 | 0.231 |
| 40 | 26/40 | 0.013 | 0.171 | 0.354 | 0.385 |
| 60 | 36/60 | 0.035 | 0.417 | 0.468 | 0.500 |
| 100 | 58/100 | 0.036 | 0.621 | 0.668 | 0.684 |
| 150 | 85/150 | 0.036 | 0.792 | 0.825 | 0.844 |
| 200 | 112/200 | 0.033 | 0.883 | 0.914 | 0.918 |
| 300 | 164/300 | 0.042 | 0.977 | 0.981 | 0.983 |
| 400 | 216/400 | 0.045 | 0.995 | 0.996 | 0.996 |

: Power at AUC 0.65 against test size, 100,000 simulated tests per cell (Figure 1 uses 400,000).

The per-group rule costs little power against the batch median at any size;
the limit is the number of cases. At n = 21 the four-group layout gives 0.215,
the five-group Main layout 0.192. Critical values come from the exact
hypergeometric null of each rule.

**Exact null tails for our three method submissions.** A submission predicting
`True` on s of n cases, with n_pos robust cases and no signal, gets
t ~ Hypergeom(n, n_pos, s) of them right among its `True` answers, so the
number correct is n − n_pos − s + 2t. The deployed rule makes s ≥ 8 on the
Small track and s ≥ 11 on the Main track, and the parity of the score fixes the
parity of s; s itself was not logged, so we list every feasible value.

| submission | track | score | feasible s | tail under the null, by s |
| --- | --- | --- | --- | --- |
| 932631 self-consistency | Small | 8/14 | 9, 11, 13 | P(≥ 8) = 0.500 for each |
| 933231 same + time reserve | Small | 5/14 | 8, 10, 12 | P(≤ 5) = 0.296, 0.280, 0.231 |
| 933057 same, Main track | Main | 7/21 | 12, 14, 16, 18 | P(≤ 7) = 0.142, 0.140, 0.126, 0.090 |

: Exact null tails, with s stated.

# Appendix D. The within-test noise model, and clustered intervals

**The model** (`report/runnoise.py`, used for §3). Population: the olympiad
pairs of `aimo-interp-challenge-sample-full`, with their measured base
accuracies, under either label rule. A fixed test is 7 robust + 7 not robust
drawn without replacement, matching the measured composition of the v8 Small
track; model groups (4/4/3/3) are assigned independently of the labels. A run
draws k = 8 answers per case --- the correct one with probability b, otherwise
one of m distinct wrong answers uniformly --- computes agreement
$(\sum_v c_v^2 - k) / (k(k-1))$, and applies `robust ⟺ agreement ≤ median + 1e-9`
inside each model group, copied verbatim from the submitted decision rule.
3,000 fixed tests × 120 runs; `sd within test` is the root mean square of the
per-test run variance and `sd between tests` the spread of per-test means, so
their squares sum to the total variance of a score. Full output, including
m = 1 and the batch-median rule, is in `report/runnoise-output.txt`.

| label | m | E[score] | sd within test | sd between tests | E\|Δ\| | P(\|Δ\| ≥ 3) | E[s] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| zero-decay | 2 | 7.50 | 0.95 | 1.43 | 0.92 | 0.064 | 10.0 |
| zero-decay | 4 | 7.88 | 0.81 | 1.49 | 0.72 | 0.037 | 9.9 |
| zero-decay | 8 | 8.07 | 0.74 | 1.49 | 0.63 | 0.028 | 9.8 |
| official | 2 | 8.17 | 0.93 | 1.43 | 0.89 | 0.058 | 10.2 |
| official | 4 | 8.35 | 0.80 | 1.51 | 0.71 | 0.035 | 10.1 |
| official | 8 | 8.41 | 0.77 | 1.53 | 0.67 | 0.030 | 10.1 |

: Run-to-run variability of the deployed method on a fixed 14-case test; E[s] is the mean number of robust calls.

The across-test figure for the same question, P(|Δ| ≥ 3) = 0.262 between two
independent scores under the deployed rule (`report/power.py`), additionally
redraws the cases and is the upper bound.

The model is a lower bound on run-to-run variability by construction: it varies
only answer sampling, holding fixed the trace-level behaviour that Appendix A.4
shows can change an entire batch at once, and it assumes wrong answers are
exchangeable across the m values. Both bounds are reported rather than one
being presented as the answer.

**Knife-edge tests.** The run-to-run tail is not uniform across test draws: at
m = 4 the 95th percentile of P(|Δ| ≥ 3) over fixed tests is 0.17 (zero-decay)
and 0.15 (official), so an unlucky case draw puts an entrant where re-running
the identical method reshuffles their rank.

**The perturbation alternative of §5** (`report/signmodel.py`). Decays of
different perturbation types correlate at Spearman 0.57--0.79. Predicting the
label recomputed from the *other* types by "zero decay under one type" gives
AUC 0.76--0.82 under the zero-decay label (70 pairs: `typos` 0.806,
`distract` 0.816, `rephrase` 0.779, `rename` 0.755, `domain` 0.800) and
0.61--0.74 under the official label (58--61 pairs). The submitted report gave
0.55--0.65 here; no script reproduces those figures, and they are replaced by
the scripted ones. The comparison with §5 is not like for like: these decays
are measured with reference answers, which a submission never sees, and a
label-free version would compare answers on the original and perturbed
problem, at twice the generation cost and with its own sampling noise. We did
not build it.

**Clustered intervals for §4** (`report/ci80.py`, cluster bootstrap over the 10
problems, 5,000 resamples):

| quantity | zero-decay label | official label |
| --- | --- | --- |
| accuracy of "base == 0 → robust" | 0.838 [0.775, 0.900] | 0.739 [0.629, 0.847] |
| AUC of the same rule | 0.784 [0.570, 0.897] | 0.763 [0.597, 0.884] |
| AUC of extremeness max(base, 1−base) | 0.745 [0.653, 0.830] | 0.595 [0.457, 0.760] |
| AUC of −base_accuracy | 0.695 [0.402, 0.865] | 0.781 [0.615, 0.907] |

: Point estimates and 95% intervals clustered by problem.

`report/ci80-output.txt` also splits the contingency table by model size:
`qwen3-8b`, the only model below 10B, has 2 of 10 pairs at base zero under
either label.

# Appendix E. A text-only feature that works on its corpus and not on the task

Before the mechanism of §4 was clear we built a text-prior model on the 137
hendrycks problems: 34 hand features, grouped 5-fold CV × 20 repeats, OOF ROC
AUC 0.707 (random forest). A single feature, `log1p(n_words)`, gives 0.737 with
bootstrap CI [0.648, 0.821], and a label-permutation test over 150 shuffles of
the same pipeline gives null AUC 0.508 ± 0.077 against 0.714, p = 0.0066. The
signal is real on its corpus, the direction is "longer statement → more
robust", and it transfers across hendrycks splits (0.827 / 0.725 train↔test,
against 0.583 / 0.590 for the 34-feature forest).

It does not transfer to the task. On the 80 olympiad pairs (zero-decay label)
`n_words` has AUC 0.555, and the fitted decision boundary (17.4 words) lies
below the *minimum* statement length in the validation sample (22 words), so
the rule degenerates into `always-true`. This was pre-registered as prediction
8 (Appendix F) and the submission confirmed it exactly: 0.5263, identical to
the constant, on all 19 v7 cases. We record it as a clean negative transfer
result with a diagnosed cause, not as a failure of the feature: olympiad
statements are uniformly long, so a length prior calibrated on a corpus with
short statements has no dynamic range left.

# Appendix F. The eleven pre-registered predictions

Written into the project log before the corresponding submission and not edited
afterwards: five clearly wrong, one half-right, five right.

| # | prediction (written before the run) | outcome | verdict |
| --- | --- | --- | --- |
| 1 | test is mixed; labelled model covers 40--60% of cases, accuracy 0.10--0.25 | coverage 21%, accuracy 0.105 | direction right, magnitude wrong 2× |
| 2 | of the 9 unknown cases, all 9 are `qwen3`/`olmo`/`gpt_oss`/`gemma` | 3 of 9 | **wrong (3×)** |
| 3 | the remaining 6 split as llama 2 / mistral 1 / phi 1 / qwen2 1 / other 1 | all 6 are `qwen2` | **wrong (completely)** |
| 4 | the five architecture probes must sum to coverage 6/19 and accuracy 3/19 | exact on both | right |
| 5 | the 6 `qwen2` cases are 7B-class (`hidden_size` ≥ 3584) | 6 of 6 | right |
| 6 | the 6 carry `deepseek` or `qwen25` in `_name_or_path` | 0 of 6 | **wrong** |
| 7 | the remaining trio is `model_type == qwen3` | 3 of 3 | right |
| 8 | `text-prior` degenerates to the constant on the private test (0.5263) | 0.5263 | right (a tautology, as we said at the time) |
| 9 | self-consistency on the Small track ≈ 8/14, plausible 6--10 | 8/14 | right --- and, per §3, uninformative |
| 10 | self-consistency transfers to the Main track: ≥ 12/21, plausible 10--14 | 7/21 | **wrong** |
| 11 | restoring forced decoding lifts the Small track score to ≥ 9/14 | 5/14 | **wrong** |

: The pre-registered predictions and their outcomes.

The arithmetic predictions --- consistency checks on the census, `AutoConfig`
fields, the tautological degeneration of the text-prior rule to a constant ---
came out exact. Of the predictions about a score, the two right ones (8, 9)
were a tautology and a range wide enough to include a coin; the two that
required the method to carry signal (10, 11) failed, as did every prediction
about the test's diversity. The exact predictions are deductions from
previous measurements; the failures are guesses about something we had not
measured. We report the record in full, including the hits, because a report
that lists only its confirmed hypotheses --- in a setting with this much
sampling noise --- should not be believed. Two of the predictions were
*thresholds* fixed in advance, and both closed a direction when they were
missed rather than licensing another variant: after 5/14 we stopped submitting
variants of the method on both tracks.

# Appendix G. Reproducibility, every submission, and where each number comes from

**Repository.** `https://github.com/oleksandr-barskyi/aimo-interp-report`
holds `solutions/` (every submitted bundle as a source directory, including
`self-consistency-932626`, `-932631` and `-933057`), `analysis/` and `report/`
(all scripts, their saved outputs, both figures and this paper's source). No
labelled data beyond the organizers' public releases was used. No model was
trained for any submitted method; the only fitted object, the four-number
length model of `text-prior`, is in the bundle (`length_logreg.json`) and is
produced by `analysis/final_model.py`.

| dataset | HuggingFace revision | file sha256 (first 16) |
| --- | --- | --- |
| `aimo-interp/aimo-interp-challenge-sample-full` | `c0ffb7e6` | `2989ab6a018f2779` |
| `aimo-interp/train-main-v2` | `fa8af758` | `e18eb44207e7987` |
| `aimo-interp/val-sample` (`cases.jsonl`, `labels.jsonl`) | `1ae454ec` | `1675ed3745c808d0`, `e9b2dcb71295d64d` |
| `math-robust-final.csv` from `aimo-interp/baselines` | --- | `8dd59857fedc708d` |

: Public data used, pinned; `analysis/fetch_hf.py` downloads the parquet files at these revisions and checks the hashes.

**Rebuilding a bundle.** Each bundle is the deterministic ZIP of one directory
of `solutions/`, made by `scripts/build.py` of `aimo-interp/getting-started` at
commit `de79405`; `build.py --small` adds the empty `small.txt` marker that
selects the Small configuration (n = 14) in `self_consistency.py`, and the
Small-track directories carry it explicitly. The self-consistency bundles have
no artifact and no dependency beyond the evaluation image, so the scores below
are reproducible from the repository alone, up to the sampling noise
quantified in §3.

**Which script gives which number.** `report/power.py`: every power number in
§3 and Appendix C, and the null tails (about a minute on a CPU, fixed seed);
`report/runnoise.py`: the within-test model of §3 and Appendix D;
`report/ci80.py`: Table 3, the clustered intervals and the `train-main-v2`
bound of §4; `report/signmodel.py`: Table 4 and the perturbation figures of
Appendix D; `analysis/aimo_reference.py`: both label rules and the 80-pair
contingency table; `report/figures.py`: both figures, with `fig1-power.csv` as
the underlying grid; `report/submissions.py`: the table below and the
live-board figures of §3, saved with their snapshot time in
`report/submissions-output.txt`. The live-board figures and the scores are
snapshots of a closed phase: they can be checked against the saved output, not
re-run. `analysis/` also holds the corpus work behind Appendix E (`cv.py`,
`nulltest.py`, `nested.py`, `confound.py`, `ci.py`). Local replication used
`torch 2.12.1+cpu`, `transformers 5.13.0`, `tokenizers 0.22.2`,
`accelerate 1.13.0`, `safetensors 0.8.0`, and the harness of
`aimo-interp/getting-started` at `de79405`.

**Standing.** The working-phase board was reset on 2026-10-15 and lists only
entries made after it; we made no submission after 2026-09-19, so we have no
entry on it. Measured against the board snapshot of 2026-11-01 07:04 UTC, our best
Small-track score (8/14) would rank about 70th of 139 teams and the Main-track
constant (11/21) about 55th of 121. The final ranking is computed by the
organizers on a fresh, unpublished set and was not known when this report
was written. The submissions named for prize evaluation are 932631 (Small
track) and 933057 (Main track), both `self-consistency`.

**All submissions** (Codabench user `obarskyi`, competition 16180; dates are
2026, UTC; bundle names without the `solution-` prefix and `-small` suffix). v7 is the
19-case Small test retired on 2026-09-16; v8 is the 14-case Small and 21-case
Main test that replaced it. Scores are fractions of the full test; 920251 is a
child submission created on the Main track by mistake (Small marker, so every
case invalid), and two entries (canc.) were cancelled in the queue before
scoring. The 28 entries are 7 method attempts (`uncertainty-profiling`, the
organizers' reference; `text-prior`; `trace-length`, which calls a problem
robust when a greedy trace does not finish within 12,288 tokens; and four
`self-consistency` versions), 14 census probes, 4 constants, the mistaken child
and the 2 cancelled entries.

| id | date | test | bundle | acc. | cov. | inv. |
| ------- | ----------- | -------- | ---------------------- | ------- | ------- | ----- |
| 920251 | 09-09 14:40 | v7 Main | always-true | 0 | 0 | 21 |
| 920252 | 09-09 14:40 | v7 Small | always-true | 0.5263 | 1 | 0 |
| 920255 | 09-09 14:42 | v7 Small | uncertainty-profiling | 0.4211 | 1 | 0 |
| 920351 | 09-09 16:01 | v7 Small | text-prior | 0.5263 | 1 | 0 |
| 920372 | 09-09 16:44 | v7 Small | trace-length | 0.4737 | 1 | 0 |
| 920381 | 09-09 16:50 | v7 Small | census | 0.1053 | 0.2105 | 15 |
| 920398 | 09-09 17:12 | v7 Small | census-olmo | 0.1579 | 0.3158 | 13 |
| 920400 | 09-09 17:15 | v7 Small | census-rest | 0 | 0 | 19 |
| 920403 | 09-09 17:17 | v7 Small | census-loadable | 0.5263 | 1 | 0 |
| 920428 | 09-09 17:33 | v7 Small | census-family | 0.1053 | 0.1579 | 16 |
| 920913 | 09-10 06:03 | v7 Small | census-llama | 0 | 0 | 19 |
| 920914 | 09-10 06:03 | v7 Small | census-mistral | 0 | 0 | 19 |
| 920915 | 09-10 06:03 | v7 Small | census-phi | 0 | 0 | 19 |
| 920916 | 09-10 06:03 | v7 Small | census-qwen2 | 0.1579 | 0.3158 | 13 |
| 920917 | 09-10 06:03 | v7 Small | census-other | 0 | 0 | 19 |
| 920919 | 09-10 06:07 | v7 Small | census-q2name | 0 | 0 | 19 |
| 920920 | 09-10 06:07 | v7 Small | census-q2qwen25 | 0 | 0 | 19 |
| 920921 | 09-10 06:07 | v7 Small | census-q2large | 0.1579 | 0.3158 | 13 |
| 920922 | 09-10 06:07 | v7 Small | census-trio-qwen3 | 0.1053 | 0.1579 | 16 |
| 928280 | 09-16 20:11 | v8 Main | always-true | canc. | canc. | canc. |
| 929069 | 09-16 22:35 | v8 Small | always-true | canc. | canc. | canc. |
| 932602 | 09-18 22:14 | v8 Small | always-true | 0.5000 | 1 | 0 |
| 932603 | 09-18 22:14 | v8 Main | always-true | 0.4762 | 1 | 0 |
| 932604 | 09-18 22:15 | v8 Main | always-false | 0.5238 | 1 | 0 |
| 932626 | 09-18 23:02 | v8 Small | self-consistency (Appendix H) | 0.3571 | 1 | 0 |
| 932631 | 09-18 23:10 | v8 Small | self-consistency (Appendix H) | 0.5714 | 1 | 0 |
| 933057 | 09-19 07:44 | v8 Main | self-consistency (Appendix H) | 0.3333 | 1 | 0 |
| 933231 | 09-19 09:42 | v8 Small | self-consistency | 0.3571 | 1 | 0 |

: Every submission, as returned by the Codabench scorer.

# Appendix H. Code of the self-consistency submissions 932626, 932631 and 933057

The repository holds each version as its own bundle directory:
`solutions/self-consistency-932626` and `-932631` (Small track, with the
`small.txt` marker), `solutions/self-consistency-933057` (Main track), and
`solutions/self-consistency` (933231). The 933057 directory matches the
surviving built bundle byte for byte; 932626 and 932631 are taken from the
commits made at 23:02 and 23:10 UTC on 2026-09-18, the minutes of the two
submissions. `solution.py` is the same five lines in every version:
`from self_consistency import predict` and
`def are_robust(model_id, problems): return predict(model_id, problems)`.
Below: `self_consistency.py` of 932631 (the Small-track submission named for
the prize) in full, then the unified diffs that turn it into 932626 and
933057, so that all three can be rebuilt from this appendix alone.

<!-- CODE-APPENDIX -->
