"""Language-neutral description of the fast-searcher behavior.

This package is evaluation/portability-only.  The active Python searcher does
not import it; parity tests compare the exported contract with the live
runtime.
"""

from agent_harness.searcher_spec.contract import (
    SEARCHER_CONTRACT_SCHEMA_RESOURCE,
    SEARCHER_CONTRACT_SCHEMA_VERSION,
    build_searcher_contract,
    dumps_searcher_contract,
    load_searcher_contract_schema,
    searcher_contract_digest,
)

__all__ = [
    "SEARCHER_CONTRACT_SCHEMA_RESOURCE",
    "SEARCHER_CONTRACT_SCHEMA_VERSION",
    "build_searcher_contract",
    "dumps_searcher_contract",
    "load_searcher_contract_schema",
    "searcher_contract_digest",
]
