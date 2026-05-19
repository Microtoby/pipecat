#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Volcengine realtime speech-to-speech service implementation.

This module provides a realtime LLM-style integration for Volcengine's
end-to-end realtime speech model. It uses Volcengine's event-driven binary
WebSocket protocol to stream user audio or text to the model and emit ASR,
LLM text, and synthesized PCM audio frames.

API reference: https://www.volcengine.com/docs/6561/1594356
"""

from __future__ import annotations

import json
import struct
import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from loguru import logger

from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    AggregationType,
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InputTextRawFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    StartFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    UserAudioRawFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService
from pipecat.services.settings import LLMSettings
from pipecat.utils.time import time_now_iso8601

try:
    from websockets.asyncio import client as websocket_client
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Volcengine, you need to `pip install pipecat-ai[volcengine]`.")
    raise Exception(f"Missing module: {e}")


VOLCENGINE_REALTIME_URL = "wss://openspeech.bytedance.com/api/v3/realtime/dialogue"
VOLCENGINE_REALTIME_RESOURCE_ID = "volc.speech.dialog"
VOLCENGINE_REALTIME_APP_KEY = "PlgvMymc7f3tQnJ6"
VOLCENGINE_REALTIME_DEFAULT_MODEL = "1.2.1.1"
VOLCENGINE_REALTIME_DEFAULT_VOICE = "zh_female_vv_jupiter_bigtts"


class _MsgType(IntEnum):
    """Volcengine binary protocol message types."""

    FULL_CLIENT_REQUEST = 0b0001
    AUDIO_ONLY_REQUEST = 0b0010
    FULL_SERVER_RESPONSE = 0b1001
    AUDIO_ONLY_RESPONSE = 0b1011
    ERROR = 0b1111


class _Flags(IntEnum):
    """Volcengine binary protocol message-type-specific flags."""

    NONE = 0b0000
    POSITIVE_SEQUENCE = 0b0001
    LAST_PACKET = 0b0010
    NEGATIVE_SEQUENCE = 0b0011
    WITH_EVENT = 0b0100


class _Serialization(IntEnum):
    """Volcengine binary protocol serialization methods."""

    NONE = 0b0000
    JSON = 0b0001


class _Compression(IntEnum):
    """Volcengine binary protocol compression methods."""

    NONE = 0b0000
    GZIP = 0b0001


class _Event(IntEnum):
    """Volcengine realtime client and server events."""

    NONE = 0

    # Client connection lifecycle.
    START_CONNECTION = 1
    FINISH_CONNECTION = 2

    # Server connection lifecycle.
    CONNECTION_STARTED = 50
    CONNECTION_FAILED = 51
    CONNECTION_FINISHED = 52

    # Client session lifecycle.
    START_SESSION = 100
    FINISH_SESSION = 102

    # Server session lifecycle.
    SESSION_STARTED = 150
    SESSION_FINISHED = 152
    SESSION_FAILED = 153
    USAGE_RESPONSE = 154

    # Client task/config events.
    TASK_REQUEST = 200
    UPDATE_CONFIG = 201
    SAY_HELLO = 300
    END_ASR = 400
    CHAT_TTS_TEXT = 500
    CHAT_TEXT_QUERY = 501
    CLIENT_INTERRUPT = 515

    # Server TTS events.
    TTS_SENTENCE_START = 350
    TTS_SENTENCE_END = 351
    TTS_RESPONSE = 352
    TTS_ENDED = 359

    # Server ASR events.
    ASR_INFO = 450
    ASR_RESPONSE = 451
    ASR_ENDED = 459

    # Server chat events.
    CHAT_RESPONSE = 550
    CHAT_TEXT_QUERY_CONFIRMED = 553
    CHAT_ENDED = 559

    # Server errors.
    DIALOG_COMMON_ERROR = 599


_CONNECTION_EVENTS = frozenset(
    {
        _Event.START_CONNECTION,
        _Event.FINISH_CONNECTION,
        _Event.CONNECTION_STARTED,
        _Event.CONNECTION_FAILED,
        _Event.CONNECTION_FINISHED,
    }
)

_SEQUENCE_FLAGS = frozenset(
    {_Flags.POSITIVE_SEQUENCE, _Flags.LAST_PACKET, _Flags.NEGATIVE_SEQUENCE}
)


@dataclass
class _Message:
    """A Volcengine event-driven binary protocol message."""

    type: int = _MsgType.FULL_CLIENT_REQUEST
    flag: int = _Flags.WITH_EVENT
    event: int = _Event.NONE
    session_id: str = ""
    serialization: int = _Serialization.JSON
    compression: int = _Compression.NONE
    payload: bytes = b""
    sequence: int | None = None
    connection_id: str = ""
    error_code: int | None = None

    def marshal(self) -> bytes:
        """Serialize the message to Volcengine's binary frame format."""
        out = bytearray(
            [
                (1 << 4) | 1,
                (int(self.type) << 4) | int(self.flag),
                (int(self.serialization) << 4) | int(self.compression),
                0x00,
            ]
        )
        if self.flag in _SEQUENCE_FLAGS:
            out += struct.pack(">i", self.sequence if self.sequence is not None else -1)
        if self.flag == _Flags.WITH_EVENT:
            out += struct.pack(">i", int(self.event))
            if self.event not in _CONNECTION_EVENTS:
                sid = self.session_id.encode("utf-8")
                out += struct.pack(">I", len(sid))
                out += sid
            elif self.type == _MsgType.FULL_SERVER_RESPONSE:
                cid = self.connection_id.encode("utf-8")
                out += struct.pack(">I", len(cid))
                out += cid
        out += struct.pack(">I", len(self.payload))
        out += self.payload
        return bytes(out)

    @classmethod
    def from_payload(
        cls,
        *,
        event: int,
        payload: dict[str, Any] | bytes | None = None,
        session_id: str = "",
        message_type: int = _MsgType.FULL_CLIENT_REQUEST,
        serialization: int | None = None,
    ) -> _Message:
        """Build an event message from a Python payload."""
        if payload is None:
            payload_bytes = b"{}"
            serialization = _Serialization.JSON if serialization is None else serialization
        elif isinstance(payload, bytes):
            payload_bytes = payload
            serialization = _Serialization.NONE if serialization is None else serialization
        else:
            payload_bytes = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
            serialization = _Serialization.JSON if serialization is None else serialization

        return cls(
            type=message_type,
            event=event,
            session_id=session_id,
            serialization=serialization,
            payload=payload_bytes,
        )

    @classmethod
    def unmarshal(cls, data: bytes) -> _Message:
        """Parse a Volcengine binary protocol frame."""
        if len(data) < 8:
            raise ValueError(f"frame too short: {len(data)} bytes")

        header_size = (data[0] & 0x0F) * 4
        msg = cls(
            type=data[1] >> 4,
            flag=data[1] & 0x0F,
            serialization=data[2] >> 4,
            compression=data[2] & 0x0F,
        )
        offset = header_size

        if msg.type == _MsgType.ERROR:
            msg.error_code = struct.unpack(">I", data[offset : offset + 4])[0]
            offset += 4

        if msg.flag in _SEQUENCE_FLAGS:
            msg.sequence = struct.unpack(">i", data[offset : offset + 4])[0]
            offset += 4

        if msg.flag == _Flags.WITH_EVENT:
            msg.event = struct.unpack(">i", data[offset : offset + 4])[0]
            offset += 4
            if msg.event in _CONNECTION_EVENTS and msg.type == _MsgType.FULL_SERVER_RESPONSE:
                size = struct.unpack(">I", data[offset : offset + 4])[0]
                offset += 4
                msg.connection_id = data[offset : offset + size].decode("utf-8", errors="replace")
                offset += size
            elif msg.event not in _CONNECTION_EVENTS:
                size = struct.unpack(">I", data[offset : offset + 4])[0]
                offset += 4
                msg.session_id = data[offset : offset + size].decode("utf-8", errors="replace")
                offset += size

        payload_size = struct.unpack(">I", data[offset : offset + 4])[0]
        offset += 4
        msg.payload = data[offset : offset + payload_size]
        if len(msg.payload) != payload_size:
            raise ValueError("payload length does not match frame header")
        return msg

    def payload_json(self) -> dict[str, Any]:
        """Decode the payload as JSON, returning an empty dict for empty payloads."""
        if not self.payload:
            return {}
        if self.serialization != _Serialization.JSON:
            return {}
        return json.loads(self.payload.decode("utf-8"))


@dataclass
class VolcengineRealtimeLLMSettings(LLMSettings):
    """Settings for VolcengineRealtimeLLMService.

    Parameters:
        voice: Volcengine realtime TTS speaker id.
        system_role: O/O2.0 model role prompt.
        speaking_style: O/O2.0 model speaking style prompt.
        character_manifest: SC/SC2.0 role manifest.
        dialog_id: Optional dialog id for continuing server-side context.
        input_mod: Input mode. Common values are ``"keep_alive"``,
            ``"push_to_talk"``, ``"text"``, and ``"audio_file"``.
        end_smooth_window_ms: Server VAD stop window in milliseconds.
        enable_custom_vad: Whether to enable custom VAD behavior.
        enable_asr_twopass: Whether to enable two-pass ASR.
        extra: Additional StartSession payload fields merged at the top level.
    """

    voice: str | None = field(default=None)
    system_role: str | None = field(default=None)
    speaking_style: str | None = field(default=None)
    character_manifest: str | None = field(default=None)
    dialog_id: str | None = field(default=None)
    input_mod: str | None = field(default=None)
    end_smooth_window_ms: int | None = field(default=None)
    enable_custom_vad: bool | None = field(default=None)
    enable_asr_twopass: bool | None = field(default=None)
    extra: dict[str, Any] = field(default_factory=dict)


class VolcengineRealtimeLLMService(LLMService):
    """Realtime speech-to-speech LLM service using Volcengine's live model.

    The service accepts raw PCM input audio and text input frames, sends them to
    Volcengine's end-to-end realtime speech model, and emits user transcripts,
    assistant text deltas, and synthesized PCM audio.
    """

    Settings = VolcengineRealtimeLLMSettings
    _settings: Settings

    def __init__(
        self,
        *,
        app_id: str,
        access_key: str,
        app_key: str = VOLCENGINE_REALTIME_APP_KEY,
        resource_id: str = VOLCENGINE_REALTIME_RESOURCE_ID,
        url: str = VOLCENGINE_REALTIME_URL,
        uid: str = "pipecat",
        input_sample_rate: int = 16000,
        output_sample_rate: int = 24000,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Volcengine realtime LLM service.

        Args:
            app_id: Volcengine APP ID, sent as ``X-Api-App-ID``.
            access_key: Volcengine Access Token, sent as ``X-Api-Access-Key``.
            app_key: Fixed realtime dialogue app key sent as ``X-Api-App-Key``.
            resource_id: Realtime dialogue resource id.
            url: Realtime dialogue WebSocket endpoint.
            uid: User identifier included in the session payload.
            input_sample_rate: Audio sample rate sent to Volcengine. The API
                expects 16 kHz mono PCM by default.
            output_sample_rate: PCM audio sample rate requested from Volcengine.
            settings: Runtime-updatable service settings.
            **kwargs: Additional arguments passed to ``LLMService``.
        """
        default_settings = self.Settings(
            model=VOLCENGINE_REALTIME_DEFAULT_MODEL,
            voice=VOLCENGINE_REALTIME_DEFAULT_VOICE,
            system_role=None,
            speaking_style=None,
            character_manifest=None,
            dialog_id=None,
            input_mod="keep_alive",
            end_smooth_window_ms=None,
            enable_custom_vad=None,
            enable_asr_twopass=None,
            extra={},
        )
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(settings=default_settings, **kwargs)

        self._app_id = app_id
        self._access_key = access_key
        self._app_key = app_key
        self._resource_id = resource_id
        self._url = url
        self._uid = uid
        self._input_sample_rate = input_sample_rate
        self._output_sample_rate = output_sample_rate

        self._socket: websocket_client.ClientConnection | None = None
        self._receive_task = None
        self._disconnecting = False
        self._session_started = False
        self._connect_id = ""
        self._session_id = ""
        self._last_user_id = ""
        self._bot_responding = False
        self._resampler = create_stream_resampler()

    def can_generate_metrics(self) -> bool:
        """Check whether this service can generate processing metrics."""
        return True

    async def start(self, frame: StartFrame):
        """Start the service and establish the realtime connection."""
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the service and close the realtime connection."""
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the service and close the realtime connection."""
        await super().cancel(frame)
        await self._disconnect()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process user audio/text frames for the realtime model."""
        await super().process_frame(frame, direction)

        if isinstance(frame, InputAudioRawFrame):
            await self._send_user_audio(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, InputTextRawFrame):
            await self._send_text_query(frame.text)
            await self.push_frame(frame, direction)
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            if frame.stop_secs:
                await self.start_ttfb_metrics(start_time=frame.timestamp - frame.stop_secs)
            await self.push_frame(frame, direction)
        elif isinstance(frame, InterruptionFrame):
            await self._send_client_interrupt()
            await self.stop_all_metrics()
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)

    async def _connect(self):
        """Open the WebSocket and initialize a realtime session."""
        if self._socket and self._socket.state is State.OPEN:
            return

        self._disconnecting = False
        self._connect_id = str(uuid.uuid4())
        self._session_id = str(uuid.uuid4())
        headers = {
            "X-Api-App-ID": self._app_id,
            "X-Api-Access-Key": self._access_key,
            "X-Api-Resource-Id": self._resource_id,
            "X-Api-App-Key": self._app_key,
            "X-Api-Connect-Id": self._connect_id,
        }

        try:
            logger.debug(f"{self} connecting to Volcengine realtime WebSocket")
            self._socket = await websocket_client.connect(self._url, additional_headers=headers)
            await self._send_message(_Message.from_payload(event=_Event.START_CONNECTION))

            reply = _Message.unmarshal(await self._socket.recv())
            if reply.event != _Event.CONNECTION_STARTED:
                raise Exception(f"expected ConnectionStarted, got {reply.event}: {reply.payload!r}")

            await self._send_message(
                _Message.from_payload(
                    event=_Event.START_SESSION,
                    session_id=self._session_id,
                    payload=self._start_session_payload(),
                )
            )

            self._receive_task = self.create_task(self._receive_messages())
            self._session_started = True
            await self._call_event_handler("on_connected")
            logger.debug(f"{self} connected to Volcengine realtime WebSocket")
        except Exception as e:
            self._socket = None
            await self.push_error(
                error_msg=f"Unable to connect to Volcengine realtime: {e}", exception=e
            )
            await self._call_event_handler("on_connection_error", f"{e}")
            raise

    async def _disconnect(self):
        """Close the active realtime session and WebSocket."""
        self._disconnecting = True
        try:
            if self._socket and self._socket.state is State.OPEN:
                if self._session_started:
                    await self._send_session_event(_Event.FINISH_SESSION, {})
                await self._send_message(_Message.from_payload(event=_Event.FINISH_CONNECTION))
                await self._socket.close()
        except Exception as e:
            logger.warning(f"{self} error disconnecting from Volcengine realtime: {e}")
        finally:
            if self._receive_task:
                await self.cancel_task(self._receive_task, timeout=1.0)
                self._receive_task = None
            self._socket = None
            self._session_started = False
            self._bot_responding = False
            await self._call_event_handler("on_disconnected")

    def _start_session_payload(self) -> dict[str, Any]:
        """Build the StartSession payload."""
        settings = self._settings
        payload: dict[str, Any] = {
            "user": {"uid": self._uid},
            "asr": {
                "audio_info": {
                    "format": "pcm",
                    "sample_rate": self._input_sample_rate,
                    "channel": 1,
                },
                "extra": {},
            },
            "tts": {
                "speaker": settings.voice or VOLCENGINE_REALTIME_DEFAULT_VOICE,
                "audio_config": {
                    "channel": 1,
                    "format": "pcm_s16le",
                    "sample_rate": self._output_sample_rate,
                },
                "extra": {},
            },
            "dialog": {
                "extra": {
                    "input_mod": settings.input_mod,
                    "model": settings.model or VOLCENGINE_REALTIME_DEFAULT_MODEL,
                }
            },
        }

        asr_extra = payload["asr"]["extra"]
        if settings.end_smooth_window_ms is not None:
            asr_extra["end_smooth_window_ms"] = settings.end_smooth_window_ms
        if settings.enable_custom_vad is not None:
            asr_extra["enable_custom_vad"] = settings.enable_custom_vad
        if settings.enable_asr_twopass is not None:
            asr_extra["enable_asr_twopass"] = settings.enable_asr_twopass

        dialog = payload["dialog"]
        for key in ("system_role", "speaking_style", "character_manifest", "dialog_id"):
            value = getattr(settings, key)
            if value is not None:
                dialog[key] = value

        payload.update(settings.extra)
        return payload

    async def _send_user_audio(self, frame: InputAudioRawFrame):
        """Send a user audio chunk to Volcengine."""
        if not self._socket or self._socket.state is not State.OPEN:
            return
        self._last_user_id = frame.user_id if isinstance(frame, UserAudioRawFrame) else ""
        audio = frame.audio
        if frame.sample_rate != self._input_sample_rate:
            audio = await self._resampler.resample(
                audio, frame.sample_rate, self._input_sample_rate
            )
        await self._send_message(
            _Message.from_payload(
                event=_Event.TASK_REQUEST,
                session_id=self._session_id,
                message_type=_MsgType.AUDIO_ONLY_REQUEST,
                serialization=_Serialization.NONE,
                payload=audio,
            )
        )

    async def _send_text_query(self, text: str):
        """Send a user text query to Volcengine."""
        if not text or not self._socket or self._socket.state is not State.OPEN:
            return
        await self._send_session_event(_Event.CHAT_TEXT_QUERY, {"content": text})
        await self.start_ttfb_metrics()

    async def _send_client_interrupt(self):
        """Notify Volcengine that the client interrupted the current response."""
        if not self._socket or self._socket.state is not State.OPEN:
            return
        await self._send_session_event(_Event.CLIENT_INTERRUPT, {})

    async def _send_session_event(self, event: int, payload: dict[str, Any]):
        """Send a JSON event scoped to the active session."""
        await self._send_message(
            _Message.from_payload(event=event, session_id=self._session_id, payload=payload)
        )

    async def _send_message(self, message: _Message):
        """Send a marshalled Volcengine message."""
        if self._disconnecting or not self._socket:
            return
        try:
            await self._socket.send(message.marshal())
        except Exception as e:
            if self._disconnecting:
                return
            await self.push_error("Volcengine realtime websocket send error", e, fatal=True)

    async def _receive_messages(self):
        """Receive and dispatch Volcengine server messages."""
        if not self._socket:
            return
        async for raw in self._socket:
            if not isinstance(raw, (bytes, bytearray)):
                logger.debug(f"{self} ignoring non-binary Volcengine message: {raw}")
                continue
            try:
                await self._handle_server_message(_Message.unmarshal(bytes(raw)))
            except Exception as e:
                if self._disconnecting:
                    return
                await self.push_error("Volcengine realtime websocket receive error", e, fatal=True)

    async def _handle_server_message(self, msg: _Message):
        """Translate a Volcengine server event into Pipecat frames."""
        if msg.type == _MsgType.ERROR:
            await self.push_error(
                error_msg=f"Volcengine realtime error {msg.error_code}: {msg.payload!r}"
            )
            return

        if msg.type == _MsgType.AUDIO_ONLY_RESPONSE or msg.event == _Event.TTS_RESPONSE:
            await self._handle_audio(msg.payload)
            return

        payload = msg.payload_json()
        event = msg.event

        if event in (_Event.CONNECTION_FAILED, _Event.SESSION_FAILED, _Event.DIALOG_COMMON_ERROR):
            await self.push_error(error_msg=f"Volcengine realtime event {event}: {payload}")
        elif event == _Event.SESSION_STARTED:
            dialog_id = payload.get("dialog_id")
            if dialog_id:
                logger.debug(f"{self} Volcengine dialog_id={dialog_id}")
        elif event == _Event.ASR_INFO:
            await self.push_frame(InterruptionFrame())
        elif event == _Event.ASR_RESPONSE:
            await self._handle_asr_response(payload)
        elif event == _Event.CHAT_RESPONSE:
            await self._handle_chat_response(payload)
        elif event == _Event.CHAT_ENDED:
            await self._handle_chat_end()
        elif event == _Event.TTS_SENTENCE_START:
            await self._handle_tts_start(payload)
        elif event == _Event.TTS_ENDED:
            await self._handle_tts_end()
        elif event == _Event.USAGE_RESPONSE:
            logger.debug(f"{self} Volcengine usage: {payload}")
        elif event in (_Event.SESSION_FINISHED, _Event.CONNECTION_FINISHED):
            logger.debug(f"{self} Volcengine realtime lifecycle event {event}")

    async def _handle_asr_response(self, payload: dict[str, Any]):
        """Emit user transcription frames from ASRResponse payloads."""
        for result in payload.get("results") or []:
            text = result.get("text")
            if not text:
                continue
            is_interim = result.get("is_interim", False)
            frame = TranscriptionFrame(
                text=text,
                user_id=self._last_user_id,
                timestamp=time_now_iso8601(),
                result=payload,
                finalized=not is_interim,
            )
            await self.push_frame(frame, FrameDirection.UPSTREAM)

    async def _handle_chat_response(self, payload: dict[str, Any]):
        """Emit assistant text deltas from ChatResponse payloads."""
        text = payload.get("content")
        if not text:
            return
        if not self._bot_responding:
            await self.start_processing_metrics()
            await self.stop_ttfb_metrics()
            await self.push_frame(LLMFullResponseStartFrame())
            self._bot_responding = True
        await self.push_frame(LLMTextFrame(text=text))
        tts_frame = TTSTextFrame(text=text, aggregated_by=AggregationType.WORD)
        tts_frame.includes_inter_frame_spaces = True
        await self.push_frame(tts_frame)

    async def _handle_chat_end(self):
        """Emit end-of-response frames when text generation ends."""
        if self._bot_responding:
            await self.stop_processing_metrics()
            await self.push_frame(LLMFullResponseEndFrame())
            self._bot_responding = False

    async def _handle_tts_start(self, payload: dict[str, Any]):
        """Emit TTS start and optional transcript text for a sentence."""
        if not self._bot_responding:
            await self.start_processing_metrics()
            await self.stop_ttfb_metrics()
            await self.push_frame(LLMFullResponseStartFrame())
            self._bot_responding = True
        await self.push_frame(TTSStartedFrame())
        text = payload.get("text")
        if text:
            frame = TTSTextFrame(text=text, aggregated_by=AggregationType.SENTENCE)
            frame.append_to_context = False
            await self.push_frame(frame)

    async def _handle_audio(self, audio: bytes):
        """Emit synthesized PCM audio from TTSResponse payloads."""
        if not audio:
            return
        if not self._bot_responding:
            await self.start_processing_metrics()
            await self.stop_ttfb_metrics()
            await self.push_frame(LLMFullResponseStartFrame())
            await self.push_frame(TTSStartedFrame())
            self._bot_responding = True
        await self.push_frame(
            TTSAudioRawFrame(audio=audio, sample_rate=self._output_sample_rate, num_channels=1)
        )

    async def _handle_tts_end(self):
        """Emit TTS and response end frames when Volcengine finishes speaking."""
        await self.push_frame(TTSStoppedFrame())
        if self._bot_responding:
            await self.stop_processing_metrics()
            await self.push_frame(LLMFullResponseEndFrame())
            self._bot_responding = False
