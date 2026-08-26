"""Test-wide guards.

Ingestion adapters cache to a gitignored directory, so a stray network call in a
test does not fail loudly: it succeeds, writes a cache file, and every later run
passes off stale data that nobody chose. Review found there was nothing stopping
that. Blocking the socket layer makes the mistake impossible rather than
unlikely.
"""

from __future__ import annotations

import socket

import pytest


class NetworkAccessDenied(RuntimeError):
    pass


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any attempt to open a socket during a test.

    Adapters must be exercised against committed fixtures. A suite that reaches
    a government API fails for reasons unrelated to the code, and quietly passes
    when that API changes shape underneath it.
    """

    def deny(*args, **kwargs):
        raise NetworkAccessDenied(
            "tests must not touch the network; parse a fixture from tests/fixtures/"
        )

    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
