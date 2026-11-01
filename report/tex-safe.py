"""Rewrite report.md's unicode symbols into LaTeX for typesetting.

report.md is kept readable in unicode; the NeurIPS template is pdflatex-based
and its fonts carry none of these glyphs, so build-pdf.sh pipes the source
through this filter first.

Substitutions are emitted as pandoc raw-LaTeX inlines wrapped in
``\\ensuremath`` rather than as ``$...$`` maths: pandoc refuses to close a
dollar-delimited span when the next character is a digit, which silently leaves
literal dollars in the output for things like "+/-1.9".  Inside `code spans`
the replacement is plain ASCII, because maths does not apply there.
"""
import re
import sys


def raw(tex: str) -> str:
    return "`" + tex + "`{=latex}"


CODE = {
    "⟺": "<=>", "≤": "<=", "≥": ">=", "→": "->",
    "−": "-", "·": "*", "×": "x", "∈": "in",
}
MATH = {
    "⁻¹": raw(r"\ensuremath{^{-1}}"),
    "≤": raw(r"\ensuremath{\leq}"), "≥": raw(r"\ensuremath{\geq}"),
    "⟺": raw(r"\ensuremath{\Leftrightarrow}"),
    "≈": raw(r"\ensuremath{\approx}"), "∈": raw(r"\ensuremath{\in}"),
    "↔": raw(r"\ensuremath{\leftrightarrow}"),
    "→": raw(r"\ensuremath{\rightarrow}"),
    "×": raw(r"\ensuremath{\times}"), "±": raw(r"\ensuremath{\pm}"),
    "−": raw(r"\ensuremath{-}"), "·": raw(r"\ensuremath{\cdot}"),
    "√": raw(r"\ensuremath{\surd}"), "Φ": raw(r"\ensuremath{\Phi}"),
    "μ": raw(r"\ensuremath{\mu}"), "Δ": raw(r"\ensuremath{\Delta}"),
    "Σ": raw(r"\ensuremath{\Sigma}"), "α": raw(r"\ensuremath{\alpha}"),
    "γ": raw(r"\ensuremath{\gamma}"),
    "ₐ": raw(r"\ensuremath{_a}"), "ᵥ": raw(r"\ensuremath{_v}"),
    "¹": raw(r"\ensuremath{^1}"), "²": raw(r"\ensuremath{^2}"),
    "§": raw(r"\S~"), "…": raw(r"\ldots{}"),
}


def convert(text: str) -> str:
    parts = re.split(r"(`[^`\n]*`)", text)
    out = []
    for i, part in enumerate(parts):
        table = CODE if i % 2 else MATH
        for k, v in table.items():
            part = part.replace(k, v)
        out.append(part)
    return "".join(out)


if __name__ == "__main__":
    sys.stdout.write(convert(sys.stdin.read()))
