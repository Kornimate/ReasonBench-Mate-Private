from diskcache import Cache

from .batching import AsyncBatcher
from .caching import AsyncCacher
from .deduplicator import AsyncDeduplicator
from .reordering import RequestReorderer
from .resources import AsyncResource
from .typedefs import Batch, BatchRequestModel, Request, Response, SingleRequestModel


class LocalAPI(AsyncResource, SingleRequestModel, BatchRequestModel):
    def __init__(
        self,
        model: BatchRequestModel,
        cache: Cache,
        batch_size: int,
        timeout: int = 30,
        allow_batch_overflow: bool = False,
    ):
        self.batcher = AsyncBatcher(
            model=model,
            batch_size=batch_size,
            timeout=timeout,
            name="local_batcher",
            allow_batch_overflow=allow_batch_overflow,
        )
        self.cacher = AsyncCacher(model=self.batcher, cache=cache)

    async def request(self, request: Request) -> Response:
        return await self.cacher.request(request)

    async def batch_request(self, batch: Batch):
        return await self.cacher.batch_request(batch)

    async def cleanup(self) -> None:
        await self.batcher.cleanup()
        await self.cacher.cleanup()


class OnlineAPI(AsyncResource, SingleRequestModel, BatchRequestModel):
    def __init__(
        self,
        model: BatchRequestModel,
        cache: Cache,
        batch_size: int,
        timeout: int = 30,
        allow_batch_overflow: bool = False,
        correctness: bool = False,
    ):
        self.cached_model = AsyncCacher(model=model, cache=cache)
        self.deduplicator = AsyncDeduplicator(model=self.cached_model, correctness=correctness)
        self.reorderer = RequestReorderer(model=self.deduplicator)
        self.batcher = AsyncBatcher(
            model=self.reorderer,
            batch_size=batch_size,
            timeout=timeout,
            name="online_batcher",
            allow_batch_overflow=allow_batch_overflow,
        )

    async def request(self, request: Request) -> Response:
        return await self.batcher.request(request)

    async def batch_request(self, batch: Batch):
        return await self.batcher.batch_request(batch)

    async def cleanup(self) -> None:
        await self.batcher.cleanup()
        await self.cached_model.cleanup()


class OrderedLocalAPI(AsyncResource, SingleRequestModel, BatchRequestModel):
    def __init__(
        self,
        model: BatchRequestModel,
        cache: Cache,
        collection_batch_size: int,
        hardware_batch_size: int,
        timeout: int = 30,
        allow_batch_overflow: bool = False,
    ):
        self.hardware_batcher = AsyncBatcher(
            model=model,
            batch_size=hardware_batch_size,
            timeout=timeout,
            name="hardware_batcher",
        )
        self.cacher = AsyncCacher(model=self.hardware_batcher, cache=cache)
        self.deduplicator = AsyncDeduplicator(model=self.cacher)
        self.reorderer = RequestReorderer(model=self.deduplicator)
        self.collection_batcher = AsyncBatcher(
            model=self.reorderer,
            batch_size=collection_batch_size,
            timeout=timeout,
            name="collection_batcher",
            allow_batch_overflow=allow_batch_overflow,
        )

    async def request(self, request: Request) -> Response:
        return await self.collection_batcher.request(request)

    async def batch_request(self, batch: Batch):
        return await self.collection_batcher.batch_request(batch)

    async def cleanup(self) -> None:
        await self.collection_batcher.cleanup()
        await self.hardware_batcher.cleanup()
        await self.cacher.cleanup()
