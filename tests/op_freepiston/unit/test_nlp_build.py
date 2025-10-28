from __future__ import annotations

from campro.freepiston.opt.nlp import build_collocation_nlp


def test_build_nlp_smoke():
    P = {"num": {"K": 5, "C": 1}}
    nlp, meta = build_collocation_nlp(P)
    assert "x" in nlp and "f" in nlp and "g" in nlp
    assert meta["K"] == 5 and meta["C"] == 1
