# Чернетка листа організаторам перед подачею (НЕ надіслано)

Кому: aimo.interp@gmail.com
Від: oleksandr.umka.barskyi@gmail.com
Вкладення: report.pdf (повний звіт)
Тема: Re: Technical report: eligibility for the write-up prize, format and submission channel

Hello AIMO Interp team,

Thank you again for your reply of 21 September. Before submitting our
technical report on OpenReview, we want to give you the full description of
the scoring-interface property we mentioned in our email of 19 September, so
that you see it from us first rather than from a reviewer.

In short: ingestion catches exceptions per model group, and scoring reports
accuracy, coverage and invalid_predictions as fractions of the full test. A
submission that answers True for the groups satisfying a predicate on
model_id and raises for the others therefore returns how many private cases
satisfy that predicate and how many of them are robust.

On 9-10 September we made fourteen such submissions on the v7 Small track
(ids 920381-920922) and used them only to count the composition of that test
by model architecture. We did not read any case-level label, did not probe
problem text, and nothing from these submissions entered any method we
submitted. The v7 test was replaced on 16 September.

Section 6 and Appendix B of the attached report describe this in full,
together with the mitigations we would suggest. We have since read the
clarification that attempts to leak evaluation data are forbidden. We leave it
to you to judge how this episode falls under that rule, and we will follow
your decision, including leaving Section 6 out of the report if you prefer.

Best regards,
Oleksandr Barskyi
Codabench: obarskyi
