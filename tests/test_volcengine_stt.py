#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import gzip
import json
from unittest.mock import AsyncMock

import pytest

from pipecat.frames.frames import InterimTranscriptionFrame, TranscriptionFrame
from pipecat.services.volcengine.stt import (
    _COMPRESSION_GZIP,
    _FLAG_HAS_SEQUENCE,
    _MSG_FULL_SERVER_RESPONSE,
    _SERIALIZATION_JSON,
    VOLCENGINE_BIGMODEL_RESOURCE_ID,
    VOLCENGINE_BIGMODEL_URL,
    VolcengineSTTService,
    _build_frame,
    _build_header,
)
from pipecat.transcriptions.language import Language


def _build_server_response(payload, *, flags=_FLAG_HAS_SEQUENCE, sequence=1):
    encoded = gzip.compress(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    header = _build_header(
        _MSG_FULL_SERVER_RESPONSE,
        flags,
        _SERIALIZATION_JSON,
        _COMPRESSION_GZIP,
    )
    return (
        header
        + sequence.to_bytes(4, "big", signed=True)
        + len(encoded).to_bytes(4, "big")
        + encoded
    )


def test_build_frame_uses_volcengine_binary_envelope():
    payload = gzip.compress(b'{"request":{"model_name":"bigmodel"}}')
    header = _build_header(1, 0, _SERIALIZATION_JSON, _COMPRESSION_GZIP)

    message = _build_frame(header, payload)

    assert message[:4] == bytes([0x11, 0x10, 0x11, 0x00])
    payload_size = int.from_bytes(message[4:8], "big")
    assert message[8 : 8 + payload_size] == payload


def test_build_request_payload_includes_settings_and_language():
    service = VolcengineSTTService(
        api_key="api-key",
        uid="user-1",
        sample_rate=16000,
        settings=VolcengineSTTService.Settings(
            language=Language.ZH_CN,
            enable_itn=True,
            enable_punc=True,
            enable_ddc=False,
        ),
    )
    service._sample_rate = 16000

    payload = service._build_request_payload()

    assert payload == {
        "user": {"uid": "user-1"},
        "audio": {
            "format": "pcm",
            "codec": "raw",
            "rate": 16000,
            "bits": 16,
            "channel": 1,
            "language": "zh-CN",
        },
        "request": {
            "model_name": "bigmodel",
            "result_type": "single",
            "show_utterances": True,
            "enable_itn": True,
            "enable_punc": True,
            "enable_ddc": False,
        },
    }


@pytest.mark.asyncio
async def test_connect_websocket_uses_new_console_api_key_headers(monkeypatch):
    service = VolcengineSTTService(api_key="api-key")
    service._sample_rate = 16000
    websocket = AsyncMock()
    websocket.send = AsyncMock()

    async def fake_websocket_connect(url, additional_headers):
        assert url == VOLCENGINE_BIGMODEL_URL
        assert additional_headers["X-Api-Key"] == "api-key"
        assert additional_headers["X-Api-Resource-Id"] == VOLCENGINE_BIGMODEL_RESOURCE_ID
        assert additional_headers["X-Api-Sequence"] == "-1"
        assert "X-Api-Request-Id" in additional_headers
        assert "X-Api-Connect-Id" in additional_headers
        assert "X-Api-App-Key" not in additional_headers
        assert "X-Api-Access-Key" not in additional_headers
        return websocket

    async def fake_call_event_handler(event_name):
        assert event_name == "on_connected"

    monkeypatch.setattr(
        "pipecat.services.volcengine.stt.websocket_connect",
        fake_websocket_connect,
    )
    monkeypatch.setattr(service, "_call_event_handler", fake_call_event_handler)

    await service._connect_websocket()

    websocket.send.assert_awaited_once()


@pytest.mark.asyncio
async def test_process_server_message_emits_interim_and_final(monkeypatch):
    service = VolcengineSTTService(
        api_key="api-key",
        settings=VolcengineSTTService.Settings(language=Language.ZH_CN),
    )
    pushed_frames = []
    traced_transcriptions = []
    stop_metrics_calls = 0

    async def fake_push_frame(frame):
        pushed_frames.append(frame)

    async def fake_handle_transcription(transcript, is_final, language=None):
        traced_transcriptions.append((transcript, is_final, language))

    async def fake_stop_processing_metrics():
        nonlocal stop_metrics_calls
        stop_metrics_calls += 1

    monkeypatch.setattr(service, "push_frame", fake_push_frame)
    monkeypatch.setattr(service, "_handle_transcription", fake_handle_transcription)
    monkeypatch.setattr(service, "stop_processing_metrics", fake_stop_processing_metrics)

    await service._process_server_message(_build_server_response({"result": {"text": "你好"}}))
    await service._process_server_message(
        _build_server_response(
            {
                "result": {
                    "text": "你好世界",
                    "utterances": [{"text": "你好世界", "definite": True}],
                }
            }
        )
    )

    assert isinstance(pushed_frames[0], InterimTranscriptionFrame)
    assert pushed_frames[0].text == "你好"
    assert isinstance(pushed_frames[1], TranscriptionFrame)
    assert pushed_frames[1].text == "你好世界"
    assert pushed_frames[1].finalized is True
    assert traced_transcriptions == [("你好世界", True, Language.ZH_CN)]
    assert stop_metrics_calls == 1
