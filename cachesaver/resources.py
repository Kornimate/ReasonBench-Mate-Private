from abc import ABC, abstractmethod
from typing import AsyncContextManager, TypeVar

T = TypeVar("T", bound="AsyncResource")


class AsyncResource(AsyncContextManager[T], ABC):
    async def __aenter__(self: T) -> T:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.cleanup()

    @abstractmethod
    async def cleanup(self) -> None:
        pass
