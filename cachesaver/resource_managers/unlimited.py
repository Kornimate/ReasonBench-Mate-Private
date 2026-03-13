from contextlib import AbstractAsyncContextManager


class UnlimitedResource:
    def __init__(self, key):
        self.data = key

    def free(self, *args, **kwargs) -> None:
        return None


class UnlimitedLimiter(AbstractAsyncContextManager):
    def __init__(self, key):
        self.key = key

    async def __aenter__(self):
        return UnlimitedResource(self.key)

    async def __aexit__(self, exc_type, exc, tb):
        return None
