# type: ignore
"""Two-node integration tests (ring + jaccl parallelism).

Run with:
    uv run pytest tests/test_2node.py -v
"""

from __future__ import annotations

import os

import pytest
from exo_tools.cluster import Thunderbolt
from exo_tools.harness import Comm, Sharding

from .framework import DEFAULT_MODEL

# GLM-5.2 model used for long-context 2-node TP tests.  Override via env if a
# different quant/variant is deployed on the cluster.
GLM_5_2_MODEL = os.environ.get("EXO_TEST_GLM_MODEL", "Jundot/GLM-5.2-oQ4")

# Token counts targeting the documented prefill→decode `Fence::wait` race
# ceiling (~45–52k agentic prompts).  These are advisory; the real prompt is
# padded to a token budget via repeated unique numbered paragraphs.
_LONG_CONTEXT_TOKEN_TARGET = int(
    os.environ.get("EXO_TEST_LONG_CONTEXT_TOKENS", "46000")
)


@pytest.mark.cluster(count=2, thunderbolt=Thunderbolt.A2A)
@pytest.mark.instance(
    DEFAULT_MODEL, sharding=Sharding.TENSOR, comm=Comm.JACCL, min_nodes=2
)
def test_2node_jaccl(session):
    resp = session.chat("Say hello in one sentence.")
    assert len(resp) > 0


@pytest.mark.cluster(count=2, thunderbolt=Thunderbolt.A2A)
@pytest.mark.instance(
    DEFAULT_MODEL, sharding=Sharding.PIPELINE, comm=Comm.RING, min_nodes=2
)
def test_2node_ring(session):
    resp = session.chat("Say hello in one sentence.")
    assert len(resp) > 0


@pytest.mark.cluster(count=2, thunderbolt=Thunderbolt.A2A)
@pytest.mark.instance(
    DEFAULT_MODEL, sharding=Sharding.TENSOR, comm=Comm.JACCL, min_nodes=2
)
def test_2node_jaccl_multi_turn(session):
    first = session.chat("What is the capital of France?")
    assert len(first) > 0
    second = session.multi_turn(
        [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": first},
            {"role": "user", "content": "What country is it in?"},
        ]
    )
    assert len(second) > 0


@pytest.mark.cluster(count=2, thunderbolt=Thunderbolt.A2A)
@pytest.mark.instance(
    GLM_5_2_MODEL, sharding=Sharding.TENSOR, comm=Comm.JACCL, min_nodes=2
)
@pytest.mark.slow
def test_2node_long_context_jaccl(session):
    """Long-context (~46k-token) agentic prompt on 2-node JACCL TP.

    Targets the prefill→decode `Fence::wait` race (exo #06 / #2208) and the
    prefill OOM ceiling (~45–52k).  Hardware-gated: requires a 2-node M3 Ultra
    cluster with the GLM-5.2 model deployed.  The decode stall watchdog
    (``EXO_DECODE_STALL_TIMEOUT``) should keep this from hanging forever even
    if the mlx collective stalls — the request fails fast instead.
    """
    # Build a long, varied prompt (~4-5 tokens per line, numbered for uniqueness
    # so the KV prefix-cache can't collapse it).  Targets the token budget.
    lines_per_chunk = 200
    chunks_needed = max(1, _LONG_CONTEXT_TOKEN_TARGET // (lines_per_chunk * 5))
    paragraphs = []
    for chunk_idx in range(chunks_needed):
        lines = [
            f"Item {chunk_idx * lines_per_chunk + line_idx}: "
            f"The quick brown fox jumps over the lazy dog number {line_idx}."
            for line_idx in range(lines_per_chunk)
        ]
        paragraphs.append("\n".join(lines))
    long_prompt = (
        "You are a helpful assistant. Here is a long document to summarize.\n\n"
        + "\n\n".join(paragraphs)
        + "\n\nIn one sentence, how many total items were listed above?"
    )

    resp = session.chat(long_prompt, max_tokens=64)
    assert len(resp) > 0
