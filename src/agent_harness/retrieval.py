"""The retrieval-client seam: the store surface the tools call, as a protocol.

Every tool that touches a Mixedbread store takes an optional ``client``. This
module types that parameter structurally so the harness can be handed a
retrieval client that is not the Mixedbread SDK. The motivating binding is a
service that runs the harness in-process: a client implementing this protocol
dispatches to the store layer already resident in the process instead of
leaving it over the public API. With the client injected, the model turn --
the caller-supplied ``generation_fn`` -- is the only network boundary a
rollout has.

The protocol is exactly what ``agent_harness.tools.functions`` calls, and no
more. It is deliberately narrow: implementations own transport, retries,
pagination and auth; the harness only reads results.

Error semantics: implementations signal provider-side failures with the
mixedbread exception types (``APIConnectionError``, ``APIStatusError`` and their
subclasses), because ``agent_harness.errors`` classifies failures by those types
and a rollout whose retrieval failed provider-side is marked as such on the
rollout record. Anything else is attributed to the agent and fed back to it as
tool feedback. mixedbread is a core dependency, so raising its exception types
costs an in-process implementation nothing.

The Mixedbread SDK client satisfies this protocol structurally: it is a valid
argument wherever a ``RetrievalClient`` is expected, and no implementation
inherits from anything here.
"""

from __future__ import annotations

import asyncio
import functools
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, TypedDict, runtime_checkable


class SearchResults(Protocol):
    """What ``Stores.search`` returns: chunk models under ``data``."""

    @property
    def data(self) -> Iterable[Any]: ...


class StoreFiles(Protocol):
    """The ``stores.files`` sub-resource."""

    def retrieve(
        self,
        *,
        file_identifier: str,
        store_identifier: str,
        return_chunks: Any,
    ) -> Any: ...

    def list(self, **kwargs: Any) -> Any: ...


class Stores(Protocol):
    """The ``stores`` resource."""

    def search(self, **kwargs: Any) -> SearchResults: ...

    def metadata_facets(self, **kwargs: Any) -> Any:
        """Aggregate metadata facets.

        The harness sends the full search-shaped request (query, top_k,
        search_options, optional facets/filters) and falls back to a
        store_identifiers-and-facets-only call on ``TypeError``, so an
        implementation that accepts only those two arguments is supported.
        """

    def list_chunks(self, **kwargs: Any) -> Any: ...

    @property
    def files(self) -> StoreFiles: ...


@runtime_checkable
class RetrievalClient(Protocol):
    """The store surface ``agent_harness`` needs to run a retrieval rollout."""

    @property
    def stores(self) -> Stores: ...

    def post(self, path: str, *, cast_to: Any, body: Mapping[str, Any]) -> Any:
        """Raw request escape hatch, used for one path only.

        ``grep`` posts to ``/v1/stores/grep``, which the SDK does not yet expose
        as a typed method; when it does, this member leaves the protocol. An
        in-process implementation only has to route that single path -- no other
        harness call reaches ``post``.
        """


class ChunkPayload(TypedDict, total=False):
    """The chunk fields the harness reads off seam results.

    Results may be pydantic models or plain mappings; either way, these are
    the keys that survive into model-visible payloads. ``store_id``,
    ``file_id`` and ``chunk_index`` form the identity key for deduplication
    and handle minting and must be stable and canonical. An in-process
    implementation can return literal ``ChunkPayload`` dicts.
    """

    id: str
    store_id: str
    file_id: str
    chunk_index: int
    score: float
    text: str
    context: str
    ocr_text: str
    transcription: str
    summary: str
    filename: str
    external_id: str
    file_title: str
    mime_type: str
    type: str
    metadata: dict[str, Any]
    generated_metadata: dict[str, Any]
    image_url: str
    media_url: str
    url: str


class StoreFilePayload(TypedDict, total=False):
    """The file fields the harness reads off ``files.retrieve`` results.

    ``chunks`` carries ``ChunkPayload``-shaped entries; ``id`` and
    ``store_id`` fall back to the requested identifiers when omitted. An
    in-process implementation can return literal ``StoreFilePayload`` dicts.
    """

    id: str
    store_id: str
    filename: str
    external_id: str
    metadata: dict[str, Any]
    status: str
    last_error: str
    mime_type: str
    type: str
    content_url: str
    chunks: list[ChunkPayload]


@dataclass(frozen=True, slots=True)
class SearchRequest:
    """One semantic search over stores."""

    query: str | None
    store_identifiers: tuple[str, ...]
    top_k: int
    return_metadata: bool = True
    filters: Mapping[str, Any] | None = None
    # False, True (server default model), or {"model": name}; parsed once per
    # process by config.search_rerank and set by search_raw, the single
    # stores.search chokepoint, so it covers every search path alike.
    rerank: bool | Mapping[str, Any] = False

    def to_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "query": self.query,
            "store_identifiers": list(self.store_identifiers),
            "top_k": self.top_k,
            "search_options": {"return_metadata": self.return_metadata},
        }
        if self.rerank:
            kwargs["search_options"]["rerank"] = self.rerank
        if self.filters:
            kwargs["filters"] = self.filters
        return kwargs


@dataclass(frozen=True, slots=True)
class MetadataFacetsRequest:
    """One metadata-facet aggregation over stores."""

    store_identifiers: tuple[str, ...]
    query: str | None = None
    top_k: int = 100
    return_metadata: bool = False
    facets: tuple[str, ...] | None = None
    filters: Mapping[str, Any] | None = None

    def to_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "query": self.query,
            "store_identifiers": list(self.store_identifiers),
            "top_k": self.top_k,
            "search_options": {"return_metadata": self.return_metadata},
        }
        if self.facets:
            kwargs["facets"] = list(self.facets)
        if self.filters:
            kwargs["filters"] = self.filters
        return kwargs

    def to_fallback_kwargs(self) -> dict[str, Any]:
        """The reduced call older SDK/service variants accept."""
        kwargs: dict[str, Any] = {"store_identifiers": list(self.store_identifiers)}
        if self.facets:
            kwargs["facets"] = list(self.facets)
        return kwargs


@dataclass(frozen=True, slots=True)
class GrepRequest:
    """One regex grep over stores."""

    store_identifiers: tuple[str, ...]
    pattern: str
    targets: tuple[str, ...]
    case_sensitive: bool
    top_k: int
    return_metadata: bool = True
    filters: Mapping[str, Any] | None = None

    def to_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "store_identifiers": list(self.store_identifiers),
            "pattern": self.pattern,
            "targets": list(self.targets),
            "case_sensitive": self.case_sensitive,
            "top_k": self.top_k,
            "return_metadata": self.return_metadata,
        }
        if self.filters:
            kwargs["filters"] = self.filters
        return kwargs


@dataclass(frozen=True, slots=True)
class ListChunksRequest:
    """One chunk listing over stores, optionally server-side sorted.

    ``sort_by`` is ``(field, ascending)``; implementations that cannot sort a
    field server-side raise ``UnprocessableEntityError`` and the harness ranks
    client-side instead.
    """

    store_identifiers: tuple[str, ...]
    top_k: int
    return_metadata: bool = True
    filters: Mapping[str, Any] | None = None
    sort_by: tuple[str, bool] | None = None

    def to_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "store_identifiers": list(self.store_identifiers),
            "top_k": self.top_k,
            "search_options": {"return_metadata": self.return_metadata},
        }
        if self.filters:
            kwargs["filters"] = self.filters
        if self.sort_by is not None:
            kwargs["sort_by"] = list(self.sort_by)
        return kwargs


@dataclass(frozen=True, slots=True)
class FileRetrieveRequest:
    """One store-file fetch, whole or as a chunk-index selection."""

    file_identifier: str
    store_identifier: str
    return_chunks: bool | tuple[int, ...]

    def to_kwargs(self) -> dict[str, Any]:
        return_chunks: Any = self.return_chunks
        if isinstance(return_chunks, tuple):
            return_chunks = list(return_chunks)
        return {
            "file_identifier": self.file_identifier,
            "store_identifier": self.store_identifier,
            "return_chunks": return_chunks,
        }


@dataclass(frozen=True, slots=True)
class FileListRequest:
    """One page of store files, optionally metadata-filtered."""

    store_identifier: str
    limit: int
    metadata_filter: Mapping[str, Any] | None = None

    def to_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "store_identifier": self.store_identifier,
            "limit": self.limit,
        }
        if self.metadata_filter:
            kwargs["metadata_filter"] = self.metadata_filter
        return kwargs


class AsyncStoreFiles(Protocol):
    """Async mirror of ``StoreFiles``, taking typed requests."""

    async def retrieve(self, request: FileRetrieveRequest) -> Any:
        """Fetch one store file; the result follows ``StoreFilePayload``."""

    async def list(self, request: FileListRequest) -> Any: ...


class AsyncStores(Protocol):
    """Async mirror of ``Stores``, taking typed requests.

    The sync protocol is kwargs-shaped because the SDK client must satisfy it
    structurally; this protocol has no such constraint -- its implementations
    are the in-process binding and ``SyncRetrievalClientAdapter``, which owns
    the kwargs flattening. That is also why grep is a first-class method here
    (the SDK routes it through its raw ``post``, handled in the adapter) and
    why the ``metadata_facets`` reduced-signature fallback for older SDK
    variants lives in the adapter rather than in this contract: an in-process
    implementation always receives the full typed request.

    ``search``/``grep``/``list_chunks`` return their hits under ``data``;
    each hit carries the ``ChunkPayload`` fields.
    """

    async def search(self, request: SearchRequest) -> Any: ...

    async def metadata_facets(self, request: MetadataFacetsRequest) -> Any: ...

    async def grep(self, request: GrepRequest) -> Any: ...

    async def list_chunks(self, request: ListChunksRequest) -> Any: ...

    @property
    def files(self) -> AsyncStoreFiles: ...


@runtime_checkable
class AsyncRetrievalClient(Protocol):
    """Async mirror of ``RetrievalClient``: the seam the agent loops run on.

    The loops are async-native; a sync ``RetrievalClient`` (the SDK included)
    participates through ``SyncRetrievalClientAdapter``, which runs each call
    on a worker thread so concurrent tool calls still overlap.
    """

    @property
    def stores(self) -> AsyncStores: ...


@dataclass(frozen=True, slots=True)
class _SyncStoreFilesAdapter:
    files: Any

    async def retrieve(self, request: FileRetrieveRequest) -> Any:
        return await asyncio.to_thread(
            functools.partial(self.files.retrieve, **request.to_kwargs())
        )

    async def list(self, request: FileListRequest) -> Any:
        return await asyncio.to_thread(functools.partial(self.files.list, **request.to_kwargs()))


@dataclass(frozen=True, slots=True)
class _SyncStoresAdapter:
    stores_resource: Any
    client: Any

    async def search(self, request: SearchRequest) -> Any:
        return await asyncio.to_thread(
            functools.partial(self.stores_resource.search, **request.to_kwargs())
        )

    async def metadata_facets(self, request: MetadataFacetsRequest) -> Any:
        # Older SDK/service variants accept only store_identifiers/facets; the
        # reduced-signature retry is this adapter's concern, never the async
        # implementation's.
        try:
            return await asyncio.to_thread(
                functools.partial(self.stores_resource.metadata_facets, **request.to_kwargs())
            )
        except TypeError:
            return await asyncio.to_thread(
                functools.partial(
                    self.stores_resource.metadata_facets, **request.to_fallback_kwargs()
                )
            )

    async def grep(self, request: GrepRequest) -> Any:
        # Prefer a typed grep once the sync client grows one; until then the
        # SDK's raw ``post`` carries the same body to the same path.
        grep_method = getattr(self.stores_resource, "grep", None)
        if callable(grep_method):
            return await asyncio.to_thread(functools.partial(grep_method, **request.to_kwargs()))
        return await asyncio.to_thread(
            functools.partial(
                self.client.post, "/v1/stores/grep", cast_to=object, body=request.to_kwargs()
            )
        )

    async def list_chunks(self, request: ListChunksRequest) -> Any:
        return await asyncio.to_thread(
            functools.partial(self.stores_resource.list_chunks, **request.to_kwargs())
        )

    @property
    def files(self) -> _SyncStoreFilesAdapter:
        return _SyncStoreFilesAdapter(self.stores_resource.files)


@dataclass(frozen=True, slots=True)
class SyncRetrievalClientAdapter:
    """``AsyncRetrievalClient`` over a sync ``RetrievalClient``.

    Every call runs on a worker thread via ``asyncio.to_thread`` so parallel
    tool calls overlap exactly as they did on the thread-pool loop. ``stores``
    resolves the underlying resource on every access rather than caching it:
    the bootstrap's failure contract probes ``client.stores`` and expects the
    injected client's own resolution error to surface, never a stale copy.
    """

    client: RetrievalClient

    @property
    def stores(self) -> _SyncStoresAdapter:
        return _SyncStoresAdapter(self.client.stores, self.client)
