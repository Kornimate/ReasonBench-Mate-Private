from typing import List

from .typedefs import Batch, BatchRequestModel, Response


class RequestReorderer(BatchRequestModel):
    def __init__(self, model: BatchRequestModel):
        self.model = model

    async def batch_request(self, batch: Batch) -> List[Response]:
        request_ids = [request.request_id for request in batch.requests]
        assert len(set(request_ids)) == len(request_ids), "Request IDs must be unique within a batch"

        original_order = {request.request_id: index for index, request in enumerate(batch.requests)}
        sorted_requests = sorted(batch.requests, key=lambda request: request.request_id)
        sorted_responses = await self.model.batch_request(Batch(requests=sorted_requests))

        reordered_responses: List[Response] = [None] * len(sorted_responses)  # type: ignore[list-item]
        for request, response in zip(sorted_requests, sorted_responses):
            reordered_responses[original_order[request.request_id]] = response

        return reordered_responses
