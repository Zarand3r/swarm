"""Step 0 smoke test: the package imports and exposes a version."""


def test_import_and_version():
    import swarm

    assert isinstance(swarm.__version__, str)
    assert swarm.__version__  # non-empty
