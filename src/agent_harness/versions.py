"""Harness version helpers for agent harness rollout records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version
from typing import Any

HARNESS_VERSION_FALLBACK = "0+unknown"

try:
    __version__ = _distribution_version("toast-harness")
except PackageNotFoundError:  # running from a source tree without install metadata
    __version__ = HARNESS_VERSION_FALLBACK

VERSION_MANIFEST_KEYS = ("harness",)


def harness_version() -> str:
    """Return the embedded harness version."""
    return __version__


def policy_version(policy_name: str | None = None) -> str:
    """Return the harness version for legacy policy-version callers."""
    del policy_name
    return harness_version()


def current_version_manifest(
    policy_name: str | None = None,
) -> dict[str, str]:
    """Return the current agent harness version manifest."""
    del policy_name
    return build_version_manifest()


def build_version_manifest(
    *,
    harness: str | None = None,
    execution_policy: str | None = None,
) -> dict[str, str]:
    """Build the structured version manifest stored on rollout metadata."""
    del execution_policy
    return {"harness": str(harness or harness_version())}


def extract_version_manifest(payload: Mapping[str, Any]) -> dict[str, str]:
    """Extract the harness version from a rollout, OpenAI trace, or metadata mapping."""
    metadata = _version_metadata(payload)
    versions = metadata.get("versions")
    if isinstance(versions, Mapping) and versions.get("harness") is not None:
        return {"harness": str(versions["harness"])}

    harness = metadata.get("harness_version")
    if harness is not None:
        return {"harness": str(harness)}
    return {}


def check_version_compatibility(
    payload: Mapping[str, Any],
    *,
    expected: Mapping[str, str] | None = None,
    required: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Check whether a rollout has required and expected harness version fields."""
    versions = extract_version_manifest(payload)
    required_keys = [str(key) for key in (required or VERSION_MANIFEST_KEYS)]
    expected_keys = (
        {str(key) for key in (expected or {})} if required is not None else set(required_keys)
    )
    missing = [key for key in required_keys if not versions.get(key)]
    mismatched = {
        str(key): {"expected": str(value), "actual": versions.get(str(key))}
        for key, value in (expected or {}).items()
        if str(key) in expected_keys
        if versions.get(str(key)) != str(value)
    }
    return {
        "compatible": not missing and not mismatched,
        "versions": versions,
        "missing": missing,
        "mismatched": mismatched,
    }


def assert_compatible_versions(
    payload: Mapping[str, Any],
    *,
    expected: Mapping[str, str] | None = None,
    required: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Return compatibility details or raise ``ValueError`` for incompatible records."""
    compatibility = check_version_compatibility(
        payload,
        expected=expected,
        required=required,
    )
    if compatibility["compatible"]:
        return compatibility
    raise ValueError(_format_compatibility_error(compatibility))


def _version_metadata(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if "versions" in payload or "harness_version" in payload:
        return payload

    openai = payload.get("openai")
    if isinstance(openai, Mapping):
        metadata = openai.get("metadata")
        if isinstance(metadata, Mapping):
            return metadata

    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        return metadata
    return {}


def _format_compatibility_error(compatibility: Mapping[str, Any]) -> str:
    details: list[str] = []
    missing = compatibility.get("missing")
    if missing:
        details.append("missing versions: " + ", ".join(str(key) for key in missing))
    mismatched = compatibility.get("mismatched")
    if isinstance(mismatched, Mapping) and mismatched:
        mismatch_details = []
        for key, values in mismatched.items():
            if isinstance(values, Mapping):
                mismatch_details.append(
                    f"{key} expected {values.get('expected')!r}, got {values.get('actual')!r}"
                )
        details.append("mismatched versions: " + "; ".join(mismatch_details))
    return "Incompatible harness rollout record (" + "; ".join(details) + ")"
