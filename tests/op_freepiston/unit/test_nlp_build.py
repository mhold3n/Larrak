from __future__ import annotations

import pytest

from campro.freepiston.opt.nlp import build_collocation_nlp


def test_build_nlp_smoke():
    P = {"num": {"K": 5, "C": 1}}
    try:
        nlp, meta = build_collocation_nlp(P)
    except RuntimeError:
        pytest.skip("CasADi not available")
        return
    assert "x" in nlp and "f" in nlp and "g" in nlp
    assert meta["K"] == 5 and meta["C"] == 1


