#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from unittest.mock import AsyncMock

import pytest

from pipecat.frames.frames import (
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
)
from pipecat.services.volcengine.realtime import (
    VOLCENGINE_REALTIME_RESOURCE_ID,
    VOLCENGINE_REALTIME_URL,
    VolcengineRealtimeLLMService,
    _Compression,
    _Event,
    _Flags,
    _Message,
    _MsgType,
    _Serialization,
)


def test_start_connection_frame_matches_documented_binary_shape():
    message = _Message.from_payload(event=_Event.START_CONNECTION).marshal()

    assert list(message) == [17, 20, 16, 0, 0, 0, 0, 1, 0, 0, 0, 2, 123, 125]


def test_session_message_roundtrip_with_json_payload():
    payload = {"dialog": {"bot_name": "豆包", "dialog_id": ""}}
    message = _Message.from_payload(
        event=_Event.START_SESSION,
        session_id="session-1",
        payload=payload,
    ).marshal()

    parsed = _Message.unmarshal(message)

    assert parsed.type == _MsgType.FULL_CLIENT_REQUEST
    assert parsed.flag == _Flags.WITH_EVENT
    assert parsed.serialization == _Serialization.JSON
    assert parsed.compression == _Compression.NONE
    assert parsed.event == _Event.START_SESSION
    assert parsed.session_id == "session-1"
    assert parsed.payload_json() == payload


def test_audio_message_uses_audio_only_request_and_raw_payload():
    message = _Message.from_payload(
        event=_Event.TASK_REQUEST,
        session_id="session-1",
        message_type=_MsgType.AUDIO_ONLY_REQUEST,
        serialization=_Serialization.NONE,
        payload=b"audio",
    ).marshal()

    parsed = _Message.unmarshal(message)

    assert parsed.type == _MsgType.AUDIO_ONLY_REQUEST
    assert parsed.event == _Event.TASK_REQUEST
    assert parsed.session_id == "session-1"
    assert parsed.serialization == _Serialization.NONE
    assert parsed.payload == b"audio"


def test_start_session_payload_contains_realtime_audio_and_model_config():
    service = VolcengineRealtimeLLMService(
        app_id="app-id",
        access_key="access-key",
        uid="user-1",
        settings=VolcengineRealtimeLLMService.Settings(
            model="1.2.1.1",
            voice="zh_female_vv_jupiter_bigtts",
            system_role="你是一个助手",
            input_mod="keep_alive",
            end_smooth_window_ms=800,
        ),
    )

    payload = service._start_session_payload()

    assert payload["user"] == {"uid": "user-1"}
    assert payload["asr"] == {
        "audio_info": {"format": "pcm", "sample_rate": 16000, "channel": 1},
        "extra": {"end_smooth_window_ms": 800},
    }
    assert payload["tts"] == {
        "speaker": "zh_female_vv_jupiter_bigtts",
        "audio_config": {"channel": 1, "format": "pcm_s16le", "sample_rate": 24000},
        "extra": {},
    }
    assert payload["dialog"] == {
        "extra": {"input_mod": "keep_alive", "model": "1.2.1.1"},
        "system_role": "你是一个助手",
    }


@pytest.mark.asyncio
async def test_connect_uses_realtime_headers_and_starts_connection(monkeypatch):
    connection_started = _Message.from_payload(
        event=_Event.CONNECTION_STARTED,
        payload={},
        message_type=_MsgType.FULL_SERVER_RESPONSE,
    ).marshal()
    websocket = AsyncMock()
    websocket.recv = AsyncMock(return_value=connection_started)
    websocket.send = AsyncMock()

    async def fake_connect(url, additional_headers):
        assert url == VOLCENGINE_REALTIME_URL
        assert additional_headers["X-Api-App-ID"] == "app-id"
        assert additional_headers["X-Api-Access-Key"] == "access-key"
        assert additional_headers["X-Api-App-Key"]
        assert additional_headers["X-Api-Resource-Id"] == VOLCENGINE_REALTIME_RESOURCE_ID
        assert "X-Api-Connect-Id" in additional_headers
        return websocket

    async def fake_call_event_handler(*args):
        pass

    service = VolcengineRealtimeLLMService(app_id="app-id", access_key="access-key")
    monkeypatch.setattr(
        "pipecat.services.volcengine.realtime.websocket_client.connect", fake_connect
    )
    monkeypatch.setattr(service, "create_task", lambda coro: coro.close())
    monkeypatch.setattr(service, "_call_event_handler", fake_call_event_handler)

    await service._connect()

    assert websocket.send.await_count == 2
    first = _Message.unmarshal(websocket.send.await_args_list[0].args[0])
    second = _Message.unmarshal(websocket.send.await_args_list[1].args[0])
    assert first.event == _Event.START_CONNECTION
    assert second.event == _Event.START_SESSION


@pytest.mark.asyncio
async def test_server_events_emit_pipecat_frames(monkeypatch):
    service = VolcengineRealtimeLLMService(app_id="app-id", access_key="access-key")
    pushed_frames = []

    async def fake_push_frame(frame, direction=None):
        pushed_frames.append(frame)

    async def noop(*args, **kwargs):
        pass

    monkeypatch.setattr(service, "push_frame", fake_push_frame)
    monkeypatch.setattr(service, "start_processing_metrics", noop)
    monkeypatch.setattr(service, "stop_processing_metrics", noop)
    monkeypatch.setattr(service, "stop_ttfb_metrics", noop)

    await service._handle_server_message(
        _Message.from_payload(
            event=_Event.ASR_RESPONSE,
            session_id="session-1",
            payload={"results": [{"text": "你好", "is_interim": False}]},
        )
    )
    await service._handle_server_message(
        _Message.from_payload(
            event=_Event.CHAT_RESPONSE,
            session_id="session-1",
            payload={"content": "你好，有什么可以帮你？"},
        )
    )
    await service._handle_server_message(
        _Message.from_payload(
            event=_Event.TTS_RESPONSE,
            session_id="session-1",
            message_type=_MsgType.AUDIO_ONLY_RESPONSE,
            serialization=_Serialization.NONE,
            payload=b"pcm",
        )
    )
    await service._handle_server_message(
        _Message.from_payload(event=_Event.TTS_ENDED, session_id="session-1", payload={})
    )

    assert any(
        isinstance(frame, TranscriptionFrame) and frame.text == "你好" for frame in pushed_frames
    )
    assert any(isinstance(frame, LLMFullResponseStartFrame) for frame in pushed_frames)
    assert any(
        isinstance(frame, LLMTextFrame) and frame.text.startswith("你好") for frame in pushed_frames
    )
    assert any(
        isinstance(frame, TTSAudioRawFrame) and frame.audio == b"pcm" for frame in pushed_frames
    )
    assert any(isinstance(frame, TTSStoppedFrame) for frame in pushed_frames)
    assert any(isinstance(frame, LLMFullResponseEndFrame) for frame in pushed_frames)
