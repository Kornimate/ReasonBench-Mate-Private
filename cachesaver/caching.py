import asyncio
from typing import Dict, List

from diskcache import Cache

from .resources import AsyncResource
from .typedefs import Batch, BatchRequestModel, Request, Response, SingleRequestModel


class AsyncCacher(AsyncResource, SingleRequestModel, BatchRequestModel):
    def __init__(self, model: SingleRequestModel, cache: Cache):
        assert cache is not None, "cache must be provided"
        assert model is not None, "model must be provided"
        assert isinstance(model, SingleRequestModel), "model must implement SingleRequestModel"

        self.model = model
        self.cache = cache
        self.key2mutex: Dict[str, asyncio.Lock] = {}
        self.namespace2used_counts: Dict[str, Dict[str, int]] = {}
        self.used_counts: Dict[str, int] = {}

    async def request(self, request: Request) -> Response:
        key = request.hash()
        mutex = self.key2mutex.get(key, asyncio.Lock())
        self.key2mutex[key] = mutex

        async with mutex:
            entries_in_cache = self.cache.get(key, [])

            if request.namespace is not None:
                used_counts = self.namespace2used_counts.setdefault(request.namespace, {})
            else:
                used_counts = self.used_counts

            used = used_counts.get(key, 0)
            num_needed = max(request.n - len(entries_in_cache[used:]), 0)

            if num_needed > 0:
                response = await self.model.request(request)
                entries_in_cache.extend(response.data)
                self.cache.set(key, entries_in_cache)

            used_counts[key] = used + request.n

            return Response(
                data=entries_in_cache[used : used + request.n],
                cached=[True] * (request.n - num_needed) + [False] * num_needed,
                duplicated=[False] * request.n,
            )

    async def batch_request(self, batch: Batch) -> List[Response]:
        return await asyncio.gather(*(self.request(request) for request in batch.requests))

    async def cleanup(self) -> None:
        self.cache.close()
