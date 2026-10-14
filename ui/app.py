
import os
import time

import chainlit as cl
from openai import AsyncOpenAI


BASE_URL = os.getenv("INFERENCE_BASE_URL", "http://localhost:8000/v1")
MODEL_NAME = os.getenv(
    "INFERENCE_MODEL",
    "Qwen/Qwen2.5-Coder-1.5B-Instruct",
)

client = AsyncOpenAI(
    base_url=BASE_URL,
    api_key=os.getenv("INFERENCE_API_KEY", "EMPTY"),
    timeout=300.0,
)

MAX_TOKENS = 512
TEMPERATURE = 0.2


@cl.on_chat_start
async def start():
    cl.user_session.set("history", [])


@cl.on_message
async def main(message: cl.Message):
    history = cl.user_session.get("history") or []

    # Do not commit the new turn to history until the request succeeds.
    messages = history + [
        {"role": "user", "content": message.content}
    ]

    response_message = cl.Message(content="")
    await response_message.send()

    assistant_text = []
    first_content_time = None
    completion_tokens = None
    start_time = time.perf_counter()

    try:
        stream = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )

        async for chunk in stream:
            # The final usage chunk may have no choices.
            if chunk.usage is not None:
                completion_tokens = chunk.usage.completion_tokens

            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta.content
            if not isinstance(delta, str) or not delta:
                continue

            if first_content_time is None:
                first_content_time = time.perf_counter()

            assistant_text.append(delta)
            await response_message.stream_token(delta)

        end_time = time.perf_counter()
        answer = "".join(assistant_text)

        # Display indicative timings, not benchmark-grade measurements.
        if first_content_time is not None:
            ttft_ms = (first_content_time - start_time) * 1000
            generation_time = end_time - first_content_time

            metrics = [
                f"TTFT: {ttft_ms:.1f} ms",
                f"Generation: {generation_time:.2f} s",
            ]

            if completion_tokens is not None:
                metrics.append(f"Output: {completion_tokens} tokens")

                if completion_tokens > 1:
                    tpot_ms = (
                        generation_time / (completion_tokens - 1)
                    ) * 1000
                    metrics.append(f"Approx. TPOT: {tpot_ms:.1f} ms")

                if generation_time > 0:
                    tok_s = completion_tokens / generation_time
                    metrics.append(f"Approx. speed: {tok_s:.1f} tok/s")

            await response_message.stream_token(
                "\n\n---\n*"
                + " | ".join(metrics)
                + "*"
            )

        # Store only the actual conversation, never the timing footer.
        history.extend(
            [
                {"role": "user", "content": message.content},
                {"role": "assistant", "content": answer},
            ]
        )
        cl.user_session.set("history", history)

        await response_message.update()

    except Exception as exc:
        # Avoid adding a failed or partial response to conversation history.
        await response_message.stream_token(
            f"\n\n**Request failed:** {type(exc).__name__}: {exc}"
        )
        await response_message.update()
