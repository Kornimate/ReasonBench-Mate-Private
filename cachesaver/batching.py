import asyncio
import time
from dataclasses import replace
from typing import List

from .resources import AsyncResource
from .typedefs import Batch, BatchRequestModel, Request, Response, SingleRequestModel


class AsyncBatcher(AsyncResource, SingleRequestModel, BatchRequestModel):
    SHUTDOWN_SIGNAL = (None, None, None)

    def __init__(
        self,
        model: BatchRequestModel,
        batch_size: int,
        timeout: float = 30,
        name: str = "unnamed",
        allow_batch_overflow: bool = False,
    ):
        self.name = name
        self.batch_size = batch_size
        self.timeout = timeout
        self.model = model
        self.allow_batch_overflow = allow_batch_overflow
        self.queue: asyncio.Queue = asyncio.Queue()
        self.worker_task = asyncio.create_task(self._batch_worker())

    async def cleanup(self) -> None:
        await self.queue.put(self.SHUTDOWN_SIGNAL)
        await self.worker_task

    async def request(self, request: Request) -> Response:
        futures = []
        for i in range(request.n):
            single_request = replace(request, n=1, request_id=f"{request.request_id}_{i}")
            future = asyncio.get_event_loop().create_future()
            item_to_put = (time.time(), single_request, future)

            if i < request.n - 1:
                self.queue.put_nowait(item_to_put)
            else:
                await self.queue.put(item_to_put)

            futures.append(future)

        responses = await asyncio.gather(*futures)
        return Response(
            data=[response.data[0] for response in responses],
            cached=[response.cached[0] if response.cached is not None else False for response in responses],
            duplicated=[
                response.duplicated[0] if response.duplicated is not None else False
                for response in responses
            ],
        )

    async def batch_request(self, batch: Batch) -> List[Response]:
        return await asyncio.gather(*(self.request(request) for request in batch.requests))

    async def _batch_worker(self) -> None:
        futures = []
        try:
            while True:
                batch_requests = []
                futures = []

                first_item = await self.queue.get()
                if first_item == self.SHUTDOWN_SIGNAL:
                    break

                first_timestamp, first_request, first_future = first_item
                batch_requests.append(first_request)
                futures.append(first_future)

                shutdown_received = False
                while len(batch_requests) < self.batch_size:
                    max_wait = max(0, self.timeout - (time.time() - first_timestamp))
                    try:
                        item = await asyncio.wait_for(self.queue.get(), timeout=max_wait)
                    except asyncio.TimeoutError:
                        break

                    if item == self.SHUTDOWN_SIGNAL:
                        shutdown_received = True
                        break

                    _, request, future = item
                    batch_requests.append(request)
                    futures.append(future)

                if self.allow_batch_overflow:
                    while True:
                        try:
                            item = self.queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break

                        if item == self.SHUTDOWN_SIGNAL:
                            shutdown_received = True
                            break

                        _, request, future = item
                        batch_requests.append(request)
                        futures.append(future)

                if batch_requests:
                    responses = await self.model.batch_request(Batch(requests=batch_requests))
                    for future, response in zip(futures, responses):
                        if not future.done():
                            future.set_result(response)

                if shutdown_received:
                    break
        except Exception as exc:
            for future in futures:
                if not future.done():
                    future.set_exception(exc)
            raise
        finally:
            while not self.queue.empty():
                item = await self.queue.get()
                if item == self.SHUTDOWN_SIGNAL:
                    continue
                _, _, future = item
                if not future.done():
                    future.cancel()
