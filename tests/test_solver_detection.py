import importlib

from campro.optimization import solver_detection as sd


def test_is_ma57_available_caches(monkeypatch):
    """`is_ma57_available` should cache its result after first call."""
    # Force fresh import to reset cache for this test session
    importlib.reload(sd)

    # Patch casadi.nlpsol to raise to simulate MA57 absence
    monkeypatch.setattr(
        "casadi.nlpsol", lambda *a, **kw: (_ for _ in ()).throw(Exception("no ma57")),
    )

    first = sd.is_ma57_available()
    second = sd.is_ma57_available()

    assert first is False  # MA57 absent
    assert second is False  # Cached result


def test_is_ma57_available_true(monkeypatch):
    """Should return True if solver creation succeeds."""
    importlib.reload(sd)

    # Fake nlpsol that succeeds
    monkeypatch.setattr("casadi.nlpsol", lambda *a, **kw: None)

    assert sd.is_ma57_available() is True
