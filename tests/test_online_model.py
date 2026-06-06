from types import SimpleNamespace

import pytest
from cachesaver.typedefs import Request

from src.models import online
from src.models.online import OnlineLLM


@pytest.mark.asyncio
async def test_online_llm_retries_until_requested_choice_count(monkeypatch):
    requested_counts = []
    returned_batches = [
        ["first", "second"],
        ["third"],
    ]

    async def fake_chat_completion(client, prompts, request, current_n, reasoning_effort=None):
        requested_counts.append(current_n)
        choices = [
            SimpleNamespace(message=SimpleNamespace(content=content))
            for content in returned_batches.pop(0)
        ]
        return SimpleNamespace(
            choices=choices,
            usage=SimpleNamespace(prompt_tokens=12, completion_tokens=9),
            model="gpt-test",
        )

    monkeypatch.setattr(online, "chat_completion", fake_chat_completion)

    model = OnlineLLM.__new__(OnlineLLM)
    model.client = object()
    model.max_n = 128
    model.reasoning_effort = None

    response = await model.request(
        Request(
            args="",
            kwargs={
                "prompt": "score this",
                "model": "gpt-test",
                "max_completion_tokens": 16,
                "temperature": 1.0,
                "top_p": 1.0,
                "stop": None,
                "logprobs": False,
            },
            n=3,
            request_id="idx0-evaluation0-state-agent0",
            namespace="test",
        )
    )

    assert [row[0] for row in response.data] == ["first", "second", "third"]
    assert requested_counts == [3, 1]
