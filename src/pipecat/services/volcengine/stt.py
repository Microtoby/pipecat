#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Volcengine (ByteDance Doubao) Speech-to-Text service implementation.

This module provides a Speech-to-Text service using Volcengine's big-model
streaming voice recognition API (大模型流式语音识别). It speaks the Volcengine
binary WebSocket protocol: gzip-compressed JSON for the initial request and
gzip-compressed raw PCM for audio packets.

API reference: https://www.volcengine.com/docs/6561/1354869
"""

import gzip
import json
import struct
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
)
from pipecat.services.settings import NOT_GIVEN, STTSettings, _NotGiven, is_given
from pipecat.services.stt_latency import VOLCENGINE_TTFS_P99
from pipecat.services.stt_service import WebsocketSTTService
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

try:
    import websockets
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Volcengine, you need to `pip install pipecat-ai[volcengine]`.")
    raise Exception(f"Missing module: {e}")

# Default WebSocket endpoint for the big-model bidirectional streaming ASR.
VOLCENGINE_BIGMODEL_URL = "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel"

# Default resource id (duration-based billing) for the big-model streaming ASR.
VOLCENGINE_BIGMODEL_RESOURCE_ID = "volc.bigasr.sauc.duration"

#
# Volcengine binary protocol constants.
#
# Every message starts with a 4-byte header:
#   byte 0: (protocol_version << 4) | header_size      (header_size in 4-byte units)
#   byte 1: (message_type << 4) | message_type_flags
#   byte 2: (serialization_method << 4) | compression
#   byte 3: reserved (0x00)
# followed by a big-endian uint32 payload size and the payload itself.
#
_PROTOCOL_VERSION = 0b0001
_HEADER_SIZE = 0b0001  # 1 * 4 = 4 bytes

_MSG_FULL_CLIENT_REQUEST = 0b0001
_MSG_AUDIO_ONLY_REQUEST = 0b0010
_MSG_FULL_SERVER_RESPONSE = 0b1001
_MSG_ERROR_RESPONSE = 0b1111

_FLAG_NONE = 0b0000
_FLAG_HAS_SEQUENCE = 0b0001
_FLAG_LAST_PACKET = 0b0010

_SERIALIZATION_NONE = 0b0000
_SERIALIZATION_JSON = 0b0001

_COMPRESSION_NONE = 0b0000
_COMPRESSION_GZIP = 0b0001


def _build_header(message_type: int, flags: int, serialization: int, compression: int) -> bytes:
    """Build the 4-byte Volcengine protocol header."""
    return bytes(
        [
            (_PROTOCOL_VERSION << 4) | _HEADER_SIZE,
            (message_type << 4) | flags,
            (serialization << 4) | compression,
            0x00,
        ]
    )


def _build_frame(header: bytes, payload: bytes) -> bytes:
    """Prepend ``header`` and a big-endian uint32 size to ``payload``."""
    return header + struct.pack(">I", len(payload)) + payload


def language_to_volcengine_language(language: Language) -> str | None:
    """Convert a Language enum to Volcengine's language code format.

    Args:
        language: The Language enum value to convert.

    Returns:
        The corresponding Volcengine language code, or the base language code
        as a fallback for unmapped languages.
    """
    LANGUAGE_MAP = {
        Language.ZH: "zh-CN",
        Language.ZH_CN: "zh-CN",
        Language.ZH_TW: "zh-TW",
        Language.ZH_HK: "zh-HK",
        Language.EN: "en-US",
        Language.EN_US: "en-US",
        Language.JA: "ja-JP",
        Language.KO: "ko-KR",
        Language.ES: "es-MX",
        Language.FR: "fr-FR",
        Language.ID: "id-ID",
        Language.VI: "vi-VN",
        Language.PT: "pt-BR",
    }
    return resolve_language(language, LANGUAGE_MAP, use_base_code=False)


@dataclass
class VolcengineSTTSettings(STTSettings):
    """Settings for VolcengineSTTService.

    ``model`` and ``language`` are inherited from ``STTSettings``. ``model`` maps
    to the Volcengine ``model_name`` request field.

    Parameters:
        enable_itn: Enable inverse text normalization (e.g. spoken numbers to digits).
        enable_punc: Enable automatic punctuation.
        enable_ddc: Enable smart disfluency removal / sentence smoothing.
    """

    enable_itn: bool | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    enable_punc: bool | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    enable_ddc: bool | _NotGiven = field(default_factory=lambda: NOT_GIVEN)


class VolcengineSTTService(WebsocketSTTService):
    """Speech-to-Text service using Volcengine's big-model streaming ASR API.

    Connects to Volcengine's bidirectional streaming WebSocket API for real-time
    transcription. Audio is streamed as gzip-compressed PCM and the service emits
    ``InterimTranscriptionFrame`` for in-progress text and ``TranscriptionFrame``
    once the server marks an utterance as definite.

    For complete API documentation, see:
    https://www.volcengine.com/docs/6561/1354869

    Event handlers:
        on_connected: Called when connected to the Volcengine service.
        on_disconnected: Called when disconnected from the Volcengine service.
        on_connection_error: Called when a connection error occurs.
    """

    Settings = VolcengineSTTSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str | None = None,
        app_key: str | None = None,
        access_key: str | None = None,
        resource_id: str = VOLCENGINE_BIGMODEL_RESOURCE_ID,
        url: str = VOLCENGINE_BIGMODEL_URL,
        uid: str = "pipecat",
        sample_rate: int | None = None,
        end_window_size: int | None = None,
        vad_segment_duration: int | None = None,
        settings: Settings | None = None,
        ttfs_p99_latency: float | None = VOLCENGINE_TTFS_P99,
        **kwargs,
    ):
        """Initialize the Volcengine STT service.

        Supports two authentication modes. Provide either ``api_key`` (new
        unified console key, sent as ``X-Api-Key``) or both ``app_key`` and
        ``access_key`` (classic speech-service credentials, sent as
        ``X-Api-App-Key`` / ``X-Api-Access-Key``).

        Args:
            api_key: Volcengine unified API key, sent as the ``X-Api-Key`` header.
            app_key: Volcengine App ID, sent as the ``X-Api-App-Key`` header.
            access_key: Volcengine Access Token, sent as the ``X-Api-Access-Key``
                header.
            resource_id: Billing resource id, sent as the ``X-Api-Resource-Id``
                header. Defaults to the duration-based 2.0 big-model ASR resource.
            url: WebSocket endpoint URL. Defaults to the optimized big-model
                streaming endpoint.
            uid: User identifier reported to the service. Defaults to ``"pipecat"``.
            sample_rate: Audio sample rate in Hz. If None, uses the pipeline
                sample rate.
            end_window_size: Forced endpoint silence window in ms (min 200, server
                default 800). Lower values finalize utterances sooner.
            vad_segment_duration: Max silence in ms before semantic segmentation
                (server default 3000).
            settings: Runtime-updatable settings. Changing them reconnects the
                service so the new configuration takes effect.
            ttfs_p99_latency: P99 latency from speech end to final transcript in
                seconds. Override for your deployment. See
                https://github.com/pipecat-ai/stt-benchmark
            **kwargs: Additional arguments passed to the parent STTService.

        Raises:
            ValueError: If neither ``api_key`` nor both ``app_key`` and
                ``access_key`` are provided.
        """
        if not api_key and not (app_key and access_key):
            raise ValueError(
                "VolcengineSTTService requires either 'api_key' or both "
                "'app_key' and 'access_key'."
            )
        default_settings = self.Settings(
            model="bigmodel",
            language=None,
            enable_itn=True,
            enable_punc=True,
            enable_ddc=False,
        )
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            sample_rate=sample_rate,
            ttfs_p99_latency=ttfs_p99_latency,
            keepalive_timeout=10,
            keepalive_interval=5,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        self._app_key = app_key
        self._access_key = access_key
        self._resource_id = resource_id
        self._url = url
        self._uid = uid
        self._end_window_size = end_window_size
        self._vad_segment_duration = vad_segment_duration

        self._receive_task = None
        self._connect_id: str = ""

    def __str__(self):
        return f"{self.name} [{self._connect_id}]"

    def can_generate_metrics(self) -> bool:
        """Check if the service can generate performance metrics.

        Returns:
            True, indicating this service supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert a pipecat Language enum to Volcengine's language code.

        Args:
            language: The Language enum value to convert.

        Returns:
            The Volcengine language code string, or None if not supported.
        """
        return language_to_volcengine_language(language)

    def _build_request_payload(self) -> dict[str, Any]:
        """Build the JSON payload for the initial full client request."""
        s = self._settings
        request: dict[str, Any] = {
            "model_name": s.model or "bigmodel",
            "result_type": "single",
            "show_utterances": True,
        }
        if is_given(s.enable_itn):
            request["enable_itn"] = s.enable_itn
        if is_given(s.enable_punc):
            request["enable_punc"] = s.enable_punc
        if is_given(s.enable_ddc):
            request["enable_ddc"] = s.enable_ddc
        if self._end_window_size is not None:
            request["end_window_size"] = self._end_window_size
        if self._vad_segment_duration is not None:
            request["vad_segment_duration"] = self._vad_segment_duration

        audio: dict[str, Any] = {
            "format": "pcm",
            "codec": "raw",
            "rate": self.sample_rate,
            "bits": 16,
            "channel": 1,
        }
        if is_given(s.language) and s.language:
            audio["language"] = self.language_to_service_language(s.language)

        return {
            "user": {"uid": self._uid},
            "audio": audio,
            "request": request,
        }

    async def start(self, frame: StartFrame):
        """Start the Volcengine STT websocket connection.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the Volcengine STT websocket connection.

        Args:
            frame: The end frame triggering service shutdown.
        """
        await super().stop(frame)
        await self._send_last_packet()
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the Volcengine STT websocket connection.

        Args:
            frame: The cancel frame triggering service cancellation.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def _update_settings(self, delta: STTSettings) -> dict[str, Any]:
        """Apply a settings delta and reconnect so the change takes effect."""
        changed = await super()._update_settings(delta)
        if changed:
            await self._request_reconnect()
        return changed

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        """Send audio data to Volcengine for transcription.

        Args:
            audio: Raw 16-bit PCM audio bytes to transcribe.

        Yields:
            None (transcription results arrive asynchronously via the WebSocket).
        """
        if self._websocket and self._websocket.state is State.OPEN:
            try:
                await self._send_audio_frame(audio, last=False)
            except websockets.exceptions.ConnectionClosed as e:
                logger.warning(f"{self} websocket closed while sending audio: {e}")
        yield None

    async def _connect(self):
        """Connect to Volcengine and start the receive task."""
        await self._connect_websocket()
        await super()._connect()
        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        """Disconnect from Volcengine and stop the receive task."""
        await super()._disconnect()
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None
        await self._disconnect_websocket()

    async def _connect_websocket(self):
        """Open the websocket and send the initial full client request."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            self._connect_id = str(uuid.uuid4())
            logger.debug(f"{self} connecting to Volcengine WebSocket")

            headers = {
                "X-Api-Resource-Id": self._resource_id,
                "X-Api-Connect-Id": self._connect_id,
                "X-Api-Request-Id": str(uuid.uuid4()),
                "X-Api-Sequence": "-1",
            }
            if self._api_key:
                headers["X-Api-Key"] = self._api_key
            else:
                headers["X-Api-App-Key"] = self._app_key
                headers["X-Api-Access-Key"] = self._access_key
            self._websocket = await websocket_connect(self._url, additional_headers=headers)

            payload = self._build_request_payload()
            compressed = gzip.compress(json.dumps(payload).encode("utf-8"))
            header = _build_header(
                _MSG_FULL_CLIENT_REQUEST, _FLAG_NONE, _SERIALIZATION_JSON, _COMPRESSION_GZIP
            )
            await self._websocket.send(_build_frame(header, compressed))

            await self._call_event_handler("on_connected")
            logger.debug(f"{self} connected to Volcengine WebSocket")
        except Exception as e:
            await self.push_error(error_msg=f"Unable to connect to Volcengine: {e}", exception=e)
            raise

    async def _disconnect_websocket(self):
        """Close the websocket connection to Volcengine."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                logger.debug(f"{self} disconnecting from Volcengine WebSocket")
                await self._websocket.close()
        except Exception as e:
            await self.push_error(error_msg=f"Error closing websocket: {e}", exception=e)
        finally:
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    async def _send_audio_frame(self, audio: bytes, last: bool):
        """Send a gzip-compressed audio-only packet."""
        flags = _FLAG_LAST_PACKET if last else _FLAG_NONE
        header = _build_header(
            _MSG_AUDIO_ONLY_REQUEST, flags, _SERIALIZATION_NONE, _COMPRESSION_GZIP
        )
        await self._websocket.send(_build_frame(header, gzip.compress(audio)))

    async def _send_last_packet(self):
        """Signal end of the audio stream with an empty last packet."""
        if self._websocket and self._websocket.state is State.OPEN:
            try:
                await self._send_audio_frame(b"", last=True)
            except Exception as e:
                logger.warning(f"{self} failed to send last packet: {e}")

    async def _send_keepalive(self, silence: bytes):
        """Send silent audio to keep the Volcengine connection alive.

        Args:
            silence: Silent 16-bit mono PCM audio bytes.
        """
        if self._websocket and self._websocket.state is State.OPEN:
            await self._send_audio_frame(silence, last=False)

    def _get_websocket(self):
        """Return the active websocket connection.

        Raises:
            Exception: If the websocket is not connected.
        """
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _receive_messages(self):
        """Receive and process Volcengine binary protocol messages."""
        async for message in self._get_websocket():
            if not isinstance(message, (bytes, bytearray)):
                continue
            await self._process_server_message(bytes(message))

    async def _process_server_message(self, data: bytes):
        """Decode a binary server message and push transcription frames."""
        if len(data) < 4:
            logger.warning(f"{self} received malformed message ({len(data)} bytes)")
            return

        message_type = (data[1] >> 4) & 0x0F
        flags = data[1] & 0x0F
        serialization = (data[2] >> 4) & 0x0F
        compression = data[2] & 0x0F
        offset = (data[0] & 0x0F) * 4

        if message_type == _MSG_ERROR_RESPONSE:
            error_code = struct.unpack(">I", data[offset : offset + 4])[0]
            offset += 4
            msg_size = struct.unpack(">I", data[offset : offset + 4])[0]
            offset += 4
            error_msg = data[offset : offset + msg_size].decode("utf-8", errors="replace")
            logger.error(f"{self} server error {error_code}: {error_msg}")
            await self.push_error(error_msg=f"Volcengine error {error_code}: {error_msg}")
            return

        if message_type != _MSG_FULL_SERVER_RESPONSE:
            logger.trace(f"{self} ignoring message type {message_type:#06b}")
            return

        if flags & _FLAG_HAS_SEQUENCE:
            offset += 4
        payload_size = struct.unpack(">I", data[offset : offset + 4])[0]
        offset += 4
        payload = data[offset : offset + payload_size]

        if compression == _COMPRESSION_GZIP:
            payload = gzip.decompress(payload)
        if serialization != _SERIALIZATION_JSON or not payload:
            return

        try:
            content = json.loads(payload)
        except json.JSONDecodeError:
            logger.warning(f"{self} received non-JSON payload")
            return

        await self._handle_result(content, is_last=bool(flags & _FLAG_LAST_PACKET))

    async def _handle_result(self, content: dict[str, Any], is_last: bool):
        """Convert a decoded server response into transcription frames."""
        result = content.get("result")
        if not result:
            return

        text = result.get("text", "")
        if not text:
            return

        utterances = result.get("utterances") or []
        definite = is_last or any(u.get("definite") for u in utterances)

        if definite:
            await self.push_frame(
                TranscriptionFrame(
                    text,
                    self._user_id,
                    time_now_iso8601(),
                    self._settings.language or None,
                    result=content,
                    finalized=True,
                )
            )
            await self._handle_transcription(text, True, self._settings.language or None)
            await self.stop_processing_metrics()
        else:
            await self.push_frame(
                InterimTranscriptionFrame(
                    text,
                    self._user_id,
                    time_now_iso8601(),
                    self._settings.language or None,
                    result=content,
                )
            )

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Language | None = None
    ):
        """Handle a transcription result for tracing."""
        pass
