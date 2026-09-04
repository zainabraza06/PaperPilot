"""XML parsing entry point.

Two of the three providers return XML, and both payloads are untrusted
input from the public internet. ``defusedxml`` disables entity expansion
and external-entity resolution (billion-laughs / XXE); we fall back to the
standard library only if it is not installed, and say so loudly.
"""

from __future__ import annotations

from typing import cast
from xml.etree.ElementTree import Element

__all__ = ["HARDENED", "Element", "parse_xml", "text_of"]

try:  # pragma: no cover - exercised by whichever branch is installed
    from defusedxml.ElementTree import fromstring as _fromstring

    HARDENED = True
except ImportError:  # pragma: no cover
    import warnings
    from xml.etree.ElementTree import fromstring as _fromstring

    HARDENED = False
    warnings.warn(
        "defusedxml is not installed; XML from upstream APIs will be parsed "
        "with the standard library, which is not hardened against entity attacks.",
        RuntimeWarning,
        stacklevel=2,
    )


def parse_xml(payload: str | bytes) -> Element:
    """Parse an XML document and return its root element."""
    # defusedxml ships no type information; this is the one place we assert it.
    return cast(Element, _fromstring(payload))


def text_of(element: Element | None) -> str | None:
    """Return an element's full text content, including nested tags' text.

    PubMed abstracts and titles contain inline markup (``<i>``, ``<sup>``),
    so ``element.text`` alone silently truncates at the first child.
    """
    if element is None:
        return None
    parts = list(element.itertext())
    joined = "".join(parts)
    collapsed = " ".join(joined.split())
    return collapsed or None
