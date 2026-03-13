from collections import defaultdict
from dataclasses import replace
from typing import List

from deepdiff import DeepHash

from .typedefs import Batch, BatchRequestModel, Request, Response


class AsyncDeduplicator(BatchRequestModel):
    def __init__(self, model: BatchRequestModel, correctness: bool = False):
        self.model = model
        self.correctness = correctness

    async def batch_request(self, batch: Batch) -> List[Response]:
        key2requests = defaultdict(list)
        key2request_and_namespace = {}

        for request in batch.requests:
            request_root = "_".join(request.request_id.split("_")[:-1]) if self.correctness else None
            request_and_namespace = (request.hash(), request.namespace, request_root)
            key = DeepHash(request_and_namespace)[request_and_namespace]
            key2request_and_namespace[key] = request_and_namespace
            key2requests[key].append(request)

        merged_requests = []
        for key in key2request_and_namespace:
            requests_to_merge = key2requests[key]
            total = sum(item.n for item in requests_to_merge)
            merged_requests.append(replace(requests_to_merge[0], n=total))

        responses = await self.model.batch_request(Batch(requests=merged_requests))
        key2results = {key: result for key, result in zip(key2request_and_namespace.keys(), responses)}

        for key, result in key2results.items():
            if result.cached:
                flags = [not flag for flag in result.cached]
                for i, flag in enumerate(flags):
                    if flag:
                        flags[i] = False
                        break
            else:
                flags = [False] + [True] * (len(result.data) - 1)
            key2results[key] = result._replace(duplicated=flags)

        results = []
        for request in batch.requests:
            request_root = "_".join(request.request_id.split("_")[:-1]) if self.correctness else None
            request_and_namespace = (request.hash(), request.namespace, request_root)
            key = DeepHash(request_and_namespace)[request_and_namespace]

            result = key2results[key]
            original_request = key2requests[key].pop(0)

            results.append(
                Response(
                    data=result.data[: original_request.n],
                    cached=result.cached[: original_request.n] if result.cached else [False] * original_request.n,
                    duplicated=result.duplicated[: original_request.n],
                )
            )

            key2results[key] = Response(
                data=result.data[original_request.n :],
                cached=result.cached[original_request.n :] if result.cached else [],
                duplicated=result.duplicated[original_request.n :],
            )

        return results
