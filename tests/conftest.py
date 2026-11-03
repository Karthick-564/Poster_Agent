import re
import shutil

import pytest

from poster_agent.spec.brand import ROOT


@pytest.fixture
def proj_tmp(request):
    """Scratch dir *inside* the project: VectorCraft is confined to the project root, so tests that render can't use tmp_path."""
    d = ROOT / "runs/_tests" / re.sub(r"[^A-Za-z0-9_]+", "_", request.node.name)
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


def pytest_collection_modifyitems(items):
    """Tests that render through VectorCraft (they use the project-local scratch dir, or skip when the CLI is missing) are 'slow'."""
    for item in items:
        uses_vc = "proj_tmp" in getattr(item, "fixturenames", ()) or any(
            "vectorcraft-cli not installed" in str(m.kwargs.get("reason", "")) for m in item.iter_markers("skipif"))
        if uses_vc:
            item.add_marker(pytest.mark.slow)


# The public repository ships no institution brand data (logos / brand profiles / real-poster fixtures). Without a brand in
# brands/<key>/, only the brand-independent tests are collected; add your own brand to run the full suite.
_BRAND_FREE = {"test_guardrails.py", "test_outputs.py"}
collect_ignore = [] if any((ROOT / "brands").glob("*/brand.yaml")) else sorted(
    p.name for p in (ROOT / "tests").glob("test_*.py") if p.name not in _BRAND_FREE)
