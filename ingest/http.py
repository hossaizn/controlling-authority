"""Shared HTTP for the ingestion adapters.

Exists for two reasons.

**Certificates.** The project pins Python 3.12 through uv, which ships its own
interpreter rather than using the system one. That interpreter does not see the
macOS trust store, so every HTTPS fetch failed with CERTIFICATE_VERIFY_FAILED
the first time the corpus was assembled end to end. Earlier fetches had run
under system Python and worked, which hid it. `certifi` makes the trust store
explicit and identical on any machine, which matters for a repo other people are
meant to be able to run.

**Duplication.** Three adapters had grown their own copy of the same urlopen
call, each with its own User-Agent and timeout.
"""

from __future__ import annotations

import ssl
import urllib.request

import certifi

USER_AGENT = "controlling-authority/0.1 (portfolio project; contact via GitHub)"
DEFAULT_TIMEOUT = 60

_SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())


def fetch(url: str, timeout: int = DEFAULT_TIMEOUT) -> bytes:
    """GET a URL with an explicit trust store. Raises on any HTTP error."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout, context=_SSL_CONTEXT) as response:
        return response.read()
