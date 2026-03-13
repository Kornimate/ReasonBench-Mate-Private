from dataclasses import asdict, dataclass, field
from typing import Any, List, NamedTuple, Optional, Protocol, runtime_checkable

from deepdiff import DeepHash


class Metadata(NamedTuple):
    n: int
    request_id: str


class PromptNamespace(NamedTuple):
    prompt: Any
    namespace: Optional[str]


@dataclass(frozen=True)
class Request:
    prompt: Any = None
    n: int = 1
    request_id: str = ""
    namespace: Optional[str] = None
    args: Any = None
    kwargs: dict[str, Any] = field(default_factory=dict)
    model: Optional[str] = None
    max_completion_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stop: Any = None
    logprobs: Optional[bool] = None
    seed: Optional[int] = None
    top_logprobs: Optional[int] = None

    def __post_init__(self) -> None:
        kwargs = dict(self.kwargs or {})
        object.__setattr__(self, "kwargs", kwargs)

        mirrored_fields = (
            "prompt",
            "model",
            "max_completion_tokens",
            "temperature",
            "top_p",
            "stop",
            "logprobs",
            "seed",
            "top_logprobs",
        )
        for field_name in mirrored_fields:
            value = getattr(self, field_name)
            if value is None and field_name in kwargs:
                object.__setattr__(self, field_name, kwargs[field_name])

        if not kwargs:
            derived_kwargs = {
                key: getattr(self, key)
                for key in mirrored_fields
                if getattr(self, key) is not None
            }
            object.__setattr__(self, "kwargs", derived_kwargs)

    def __getattr__(self, name: str) -> Any:
        if name in self.kwargs:
            return self.kwargs[name]
        return None

    def hash(self) -> int:
        params = asdict(self)
        for key in ("request_id", "namespace", "n"):
            params.pop(key, None)

        # Prefer the structured kwargs payload when present so request hashes stay
        # stable across the repo's legacy and zip-derived calling styles.
        if params.get("kwargs"):
            for key in (
                "prompt",
                "model",
                "max_completion_tokens",
                "temperature",
                "top_p",
                "stop",
                "logprobs",
                "seed",
                "top_logprobs",
            ):
                params.pop(key, None)

        return DeepHash(params)[params]


class Batch(NamedTuple):
    requests: List[Request]


class Response(NamedTuple):
    data: List[Any]
    cached: Optional[List[bool]] = None
    duplicated: Optional[List[bool]] = None


@runtime_checkable
class SingleRequestModel(Protocol):
    async def request(self, request: Request) -> Response: ...


@runtime_checkable
class BatchRequestModel(Protocol):
    async def batch_request(self, batch: Batch) -> List[Response]: ...
