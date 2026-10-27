"""Canonical serialization and content identities for experiment artifacts."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def content_sha256(value: Any) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_id_hash(values: list[str] | tuple[str, ...]) -> str:
    """Hash an identity set independently of input row order."""

    normalized = sorted(map(str, values))
    if len(normalized) != len(set(normalized)):
        raise ValueError("identity hash requires unique ids")
    return sha256("\n".join(normalized).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- forward fix
# Phase 3.4E finding A6: manifests hashed working-tree bytes, which on a Windows
# checkout with core.autocrlf=true were CRLF for most text files while git stores
# LF. From Phase 3.4 COMMIT A onward, text artifacts are hashed over their
# LF-normalised bytes so the value equals the git blob hash on every platform.
TEXT_SUFFIXES = frozenset({".json", ".jsonl", ".csv", ".md", ".py", ".txt", ".yaml", ".yml"})


def canonical_bytes(path: str | Path) -> bytes:
    """Bytes of ``path`` with CRLF normalised to LF for text artifacts, raw otherwise."""

    path = Path(path)
    raw = path.read_bytes()
    if path.suffix.lower() in TEXT_SUFFIXES:
        return raw.replace(b"\r\n", b"\n")
    return raw


def canonical_sha256(path: str | Path) -> str:
    """Portable content hash: equals ``file_sha256`` of the LF form / the git blob."""

    return sha256(canonical_bytes(path)).hexdigest()


def hash_form(path: str | Path, recorded: str) -> str:
    """Classify a recorded hash as LF, CRLF or NEITHER relative to ``path``."""

    lf = canonical_bytes(path)
    if recorded == sha256(lf).hexdigest():
        return "LF"
    if Path(path).suffix.lower() in TEXT_SUFFIXES and \
            recorded == sha256(lf.replace(b"\n", b"\r\n")).hexdigest():
        return "CRLF"
    return "NEITHER"


def initialization_hash(model) -> str:
    """SHA-256 over the concatenated initial trainable parameters, in module order.

    Taken immediately after construction and before the first optimizer step so
    that paired PRESENT/ABSENT models seeded identically can be shown to start
    from byte-identical weights (Phase 3.4E finding A4).
    """

    import numpy as np

    digest = sha256()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        digest.update(name.encode("utf-8"))
        digest.update(np.ascontiguousarray(
            parameter.detach().cpu().numpy(), dtype=np.float32).tobytes())
    return digest.hexdigest()


def initial_state_hash(model) -> str:
    """SHA-256 over the COMPLETE initial ``state_dict`` (Phase 3.4F.1 finding F2).

    Covers trainable parameters and every non-trainable buffer - BatchNorm
    ``running_mean``, ``running_var`` and ``num_batches_tracked`` included.
    Canonical ordering is ``state_dict()`` insertion order (module registration
    order, which is deterministic for a fixed architecture). For each entry the
    digest absorbs the key, the dtype string, the shape tuple and the raw tensor
    bytes in the tensor's own dtype, so serialization metadata and pickle
    framing never enter the hash.
    """

    import numpy as np

    digest = sha256()
    for key, tensor in model.state_dict().items():
        array = np.ascontiguousarray(tensor.detach().cpu().numpy())
        digest.update(key.encode("utf-8"))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(str(tuple(array.shape)).encode("utf-8"))
        digest.update(array.tobytes())
    return digest.hexdigest()
