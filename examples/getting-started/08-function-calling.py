#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.volcengine.realtime import VolcengineRealtimeLLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

load_dotenv(override=True)


# We use lambdas to defer transport parameter creation until the transport
# type is selected at runtime.
transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")

    # Volcengine's end-to-end realtime speech model handles speech recognition,
    # the LLM, and speech synthesis in a single service. The realtime API does
    # not support custom function/tool calling; instead we enable the model's
    # built-in web search so it can still answer questions about real-time
    # information such as weather or news. Web search requires a Volcengine
    # 融合信息搜索 (fused information search) API key — set
    # VOLCENGINE_WEBSEARCH_API_KEY to enable it.
    websearch_api_key = os.environ.get("VOLCENGINE_WEBSEARCH_API_KEY")

    llm = VolcengineRealtimeLLMService(
        app_id=os.environ["VOLCENGINE_APP_ID"],
        access_key=os.environ["VOLCENGINE_ACCESS_TOKEN"],
        enable_websearch=bool(websearch_api_key),
        websearch_api_key=websearch_api_key,
        greeting="你好，我是你的语音助手，很高兴和你聊天。有什么可以帮你的吗？",
        settings=VolcengineRealtimeLLMService.Settings(
            system_role=(
                "你是一位耐心的陪聊助手，你正在通过电话和用户交谈，避免输出表情和任何不能被读出的"
                "内容，你应该给予用户更多情绪价值，迎合他们的话语，但不是一个全知全能的AI。"
            ),
        ),
    )

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            user_aggregator,
            llm,
            transport.output(),
            assistant_aggregator,
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # The bot's opening greeting is sent automatically by the realtime
        # service once its session starts (see the `greeting` argument above).

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
