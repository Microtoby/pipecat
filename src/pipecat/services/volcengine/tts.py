#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Volcengine (ByteDance Doubao) Text-to-Speech service implementation.

This module provides a Text-to-Speech service using Volcengine's big-model
bidirectional streaming voice synthesis API (大模型语音合成 双向流式). It
speaks the Volcengine event-driven binary WebSocket protocol.

API reference: https://www.volcengine.com/docs/6561/1719100
"""

import asyncio
import json
import struct
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
)
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import WebsocketTTSService
from pipecat.transcriptions.language import Language

try:
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Volcengine, you need to `pip install pipecat-ai[volcengine]`.")
    raise Exception(f"Missing module: {e}")

# Default WebSocket endpoint for the big-model bidirectional streaming TTS.
VOLCENGINE_TTS_URL = "wss://openspeech.bytedance.com/api/v3/tts/bidirection"

# Default resource id for the big-model bidirectional streaming TTS.
VOLCENGINE_TTS_RESOURCE_ID = "volc.service_type.10029"

# Default speaker (a big-model Chinese female voice paired with the default
# resource id above). Voices must match their resource id — e.g. the
# ``*_uranus_bigtts`` Seed-TTS voices require ``resource_id="seed-tts-2.0"``.
VOLCENGINE_TTS_DEFAULT_VOICE = "zh_female_cancan_mars_bigtts"


class _MsgType(IntEnum):
    """Volcengine binary protocol message types (high 4 bits of header byte 1)."""

    FULL_CLIENT_REQUEST = 0b0001
    AUDIO_ONLY_CLIENT = 0b0010
    FULL_SERVER_RESPONSE = 0b1001
    AUDIO_ONLY_SERVER = 0b1011
    ERROR = 0b1111


class _Flags(IntEnum):
    """Volcengine binary protocol message-type-specific flags (low 4 bits of byte 1)."""

    NONE = 0b0000
    POSITIVE_SEQ = 0b0001
    LAST_NO_SEQ = 0b0010
    NEGATIVE_SEQ = 0b0011
    WITH_EVENT = 0b0100


class _Event(IntEnum):
    """Volcengine bidirectional streaming TTS protocol events."""

    NONE = 0
    # Upstream connection lifecycle
    START_CONNECTION = 1
    FINISH_CONNECTION = 2
    # Downstream connection lifecycle
    CONNECTION_STARTED = 50
    CONNECTION_FAILED = 51
    CONNECTION_FINISHED = 52
    # Upstream session lifecycle
    START_SESSION = 100
    CANCEL_SESSION = 101
    FINISH_SESSION = 102
    # Downstream session lifecycle
    SESSION_STARTED = 150
    SESSION_CANCELED = 151
    SESSION_FINISHED = 152
    SESSION_FAILED = 153
    # Upstream task
    TASK_REQUEST = 200
    # Downstream TTS results
    TTS_SENTENCE_START = 350
    TTS_SENTENCE_END = 351
    TTS_RESPONSE = 352
    TTS_ENDED = 359


# Events that do NOT carry a session id on the wire.
_CONNECTION_EVENTS = frozenset(
    {
        _Event.START_CONNECTION,
        _Event.FINISH_CONNECTION,
        _Event.CONNECTION_STARTED,
        _Event.CONNECTION_FAILED,
        _Event.CONNECTION_FINISHED,
    }
)

_SERIALIZATION_JSON = 0b0001
_COMPRESSION_NONE = 0b0000


@dataclass
class _Message:
    """A Volcengine binary protocol message (event-driven framing)."""

    type: int = _MsgType.FULL_CLIENT_REQUEST
    flag: int = _Flags.WITH_EVENT
    event: int = _Event.NONE
    session_id: str = ""
    error_code: int = 0
    payload: bytes = b""

    def marshal(self) -> bytes:
        """Serialize the message to bytes for sending to the server."""
        out = bytearray(
            [
                (1 << 4) | 1,  # protocol version 1, header size 1 (4 bytes)
                (int(self.type) << 4) | int(self.flag),
                (_SERIALIZATION_JSON << 4) | _COMPRESSION_NONE,
                0x00,
            ]
        )
        if self.flag == _Flags.WITH_EVENT:
            out += struct.pack(">i", int(self.event))
            if self.event not in _CONNECTION_EVENTS:
                sid = self.session_id.encode("utf-8")
                out += struct.pack(">I", len(sid))
                out += sid
        out += struct.pack(">I", len(self.payload))
        out += self.payload
        return bytes(out)

    @classmethod
    def unmarshal(cls, data: bytes) -> "_Message":
        """Parse a binary message received from the server.

        Raises:
            ValueError: If the frame is too short or malformed.
        """
        if len(data) < 4:
            raise ValueError(f"frame too short: {len(data)} bytes")

        msg = cls(type=data[1] >> 4, flag=data[1] & 0x0F)
        offset = (data[0] & 0x0F) * 4  # header size in bytes

        if msg.flag == _Flags.WITH_EVENT:
            msg.event = struct.unpack(">i", data[offset : offset + 4])[0]
            offset += 4
            if msg.event not in _CONNECTION_EVENTS:
                sid_size = struct.unpack(">I", data[offset : offset + 4])[0]
                offset += 4
                if sid_size:
                    msg.session_id = data[offset : offset + sid_size].decode(
                        "utf-8", errors="replace"
                    )
                    offset += sid_size
            elif msg.type == _MsgType.FULL_SERVER_RESPONSE:
                # Server connection events carry a connect id (size + bytes).
                cid_size = struct.unpack(">I", data[offset : offset + 4])[0]
                offset += 4 + cid_size

        if msg.type == _MsgType.ERROR:
            msg.error_code = struct.unpack(">I", data[offset : offset + 4])[0]
            offset += 4

        payload_size = struct.unpack(">I", data[offset : offset + 4])[0]
        offset += 4
        msg.payload = data[offset : offset + payload_size]
        return msg


@dataclass
class VolcengineTTSSettings(TTSSettings):
    """Settings for VolcengineTTSService.

    ``voice`` (inherited from ``TTSSettings``) is the Volcengine speaker id.
    ``model`` and ``language`` are inherited but unused — the model is selected
    by ``resource_id`` and ``voice``.

    Parameters:
        speed: Speech rate multiplier. ``1.0`` is normal speed; the range
            ``[0.5, 2.0]`` maps to Volcengine's integer ``speech_rate``
            percent-delta ``[-50, 100]``.
    """

    speed: float | None = field(default=None)


class VolcengineTTSService(WebsocketTTSService):
    """Text-to-Speech service using Volcengine's bidirectional streaming TTS API.

    Connects to Volcengine's event-driven bidirectional WebSocket API and emits
    ``TTSAudioRawFrame`` audio as it is synthesized. Each turn maps to a
    Volcengine session; the pipeline ``context_id`` is used as the session id.
    On interruption the active session is cancelled (via ``CancelSession``) while
    the WebSocket connection is kept open for the next turn.

    Supports two authentication modes. Provide either ``api_key`` (sent as
    ``X-Api-Key``) or both ``app_key`` and ``access_key`` (sent as
    ``X-Api-App-Key`` / ``X-Api-Access-Key``).

    For complete API documentation, see:
    https://www.volcengine.com/docs/6561/1719100

    Event handlers:
        on_connected: Called when connected to the Volcengine service.
        on_disconnected: Called when disconnected from the Volcengine service.
        on_connection_error: Called when a connection error occurs.
    """

    Settings = VolcengineTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str | None = None,
        app_key: str | None = None,
        access_key: str | None = None,
        resource_id: str = VOLCENGINE_TTS_RESOURCE_ID,
        url: str = VOLCENGINE_TTS_URL,
        voice: str = VOLCENGINE_TTS_DEFAULT_VOICE,
        uid: str = "pipecat",
        sample_rate: int | None = None,
        emotion: str | None = None,
        emotion_scale: int | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Volcengine TTS service.

        Args:
            api_key: Volcengine unified API key, sent as the ``X-Api-Key`` header.
            app_key: Volcengine App ID, sent as the ``X-Api-App-Key`` header.
            access_key: Volcengine Access Token, sent as the ``X-Api-Access-Key``
                header.
            resource_id: Billing resource id, sent as the ``X-Api-Resource-Id``
                header. Defaults to the big-model TTS resource.
            url: WebSocket endpoint URL. Defaults to the bidirectional streaming
                endpoint.
            voice: Volcengine speaker id. Must be compatible with ``resource_id``
                (e.g. ``"zh_female_cancan_mars_bigtts"`` for ``volc.service_type.10029``).
            uid: User identifier reported to the service. Defaults to ``"pipecat"``.
            sample_rate: Output audio sample rate in Hz. If None, uses the pipeline
                sample rate.
            emotion: Optional emotion preset (e.g. ``"happy"``, ``"sad"``,
                ``"angry"``) for emotion-capable voices.
            emotion_scale: Optional emotion intensity (1-5) when ``emotion`` is set.
            settings: Runtime-updatable settings (voice, speed).
            **kwargs: Additional arguments passed to the parent TTS service.

        Raises:
            ValueError: If neither ``api_key`` nor both ``app_key`` and
                ``access_key`` are provided.
        """
        if not api_key and not (app_key and access_key):
            raise ValueError(
                "VolcengineTTSService requires either 'api_key' or both 'app_key' and 'access_key'."
            )

        default_settings = self.Settings(
            voice=voice,
            language=None,
            speed=None,
        )
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            push_start_frame=True,
            push_text_frames=True,
            sample_rate=sample_rate,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        self._app_key = app_key
        self._access_key = access_key
        self._resource_id = resource_id
        self._url = url
        self._uid = uid
        self._emotion = emotion
        self._emotion_scale = emotion_scale

        self._receive_task = None
        # context ids for which a Volcengine session has been started.
        self._active_sessions: set[str] = set()

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as the Volcengine TTS service supports metrics generation.
        """
        return True

    def _speech_rate(self) -> int:
        """Convert the ``speed`` multiplier to Volcengine's integer percent-delta."""
        speed = self._settings.speed
        if not speed:
            return 0
        return max(-50, min(100, round((speed - 1.0) * 100)))

    def _req_params(self, text: str | None = None) -> dict[str, Any]:
        """Build the ``req_params`` block for a session/task request."""
        audio_params: dict[str, Any] = {
            "format": "pcm",
            "sample_rate": self.sample_rate,
            "speech_rate": self._speech_rate(),
        }
        if self._emotion:
            audio_params["emotion"] = self._emotion
        if self._emotion_scale is not None:
            audio_params["emotion_scale"] = self._emotion_scale

        params: dict[str, Any] = {
            "speaker": self._settings.voice,
            "audio_params": audio_params,
        }
        if text is not None:
            params["text"] = text
        return params

    async def start(self, frame: StartFrame):
        """Start the Volcengine TTS service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the Volcengine TTS service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the Volcengine TTS service.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def _connect(self):
        """Connect to Volcengine and start the receive task."""
        await super()._connect()
        await self._connect_websocket()
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
        """Open the websocket and complete the StartConnection handshake."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            logger.debug(f"{self} connecting to Volcengine TTS WebSocket")
            headers = {
                "X-Api-Resource-Id": self._resource_id,
                "X-Api-Connect-Id": str(uuid.uuid4()),
                "X-Api-Request-Id": str(uuid.uuid4()),
            }
            if self._api_key:
                headers["X-Api-Key"] = self._api_key
            else:
                headers["X-Api-App-Id"] = self._app_key
                headers["X-Api-App-Key"] = self._app_key
                headers["X-Api-Access-Key"] = self._access_key

            self._websocket = await websocket_connect(self._url, additional_headers=headers)
            self._active_sessions.clear()

            # StartConnection handshake — wait for ConnectionStarted before
            # starting any sessions.
            await self._websocket.send(
                _Message(event=_Event.START_CONNECTION, payload=b"{}").marshal()
            )
            reply = _Message.unmarshal(await asyncio.wait_for(self._websocket.recv(), timeout=10))
            if reply.event != _Event.CONNECTION_STARTED:
                raise Exception(
                    f"expected ConnectionStarted, got event {reply.event}: {reply.payload!r}"
                )

            await self._call_event_handler("on_connected")
            logger.debug(f"{self} connected to Volcengine TTS WebSocket")
        except Exception as e:
            await self.push_error(
                error_msg=f"Unable to connect to Volcengine TTS: {e}", exception=e
            )
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        """Close the websocket connection to Volcengine."""
        try:
            await self.stop_all_metrics()
            if self._websocket:
                logger.debug(f"{self} disconnecting from Volcengine TTS WebSocket")
                await self._websocket.send(
                    _Message(event=_Event.FINISH_CONNECTION, payload=b"{}").marshal()
                )
                await self._websocket.close()
        except Exception as e:
            logger.warning(f"{self} error disconnecting: {e}")
        finally:
            self._websocket = None
            self._active_sessions.clear()
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        """Return the active websocket connection.

        Raises:
            Exception: If the websocket is not connected.
        """
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _send_session_message(self, event: int, session_id: str, payload: dict[str, Any]):
        """Send an event-carrying client message for a given session."""
        body = dict(payload, event=int(event))
        message = _Message(
            event=event,
            session_id=session_id,
            payload=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        )
        await self._get_websocket().send(message.marshal())

    async def flush_audio(self, context_id: str | None = None):
        """Finish the Volcengine session so the server finalizes the audio.

        Args:
            context_id: The context (session) to flush. If None, falls back to
                the currently active context.
        """
        session_id = context_id or self.get_active_audio_context_id()
        if not session_id or session_id not in self._active_sessions:
            return
        if not self._websocket or self._websocket.state is not State.OPEN:
            return
        logger.trace(f"{self} finishing Volcengine session {session_id}")
        await self._send_session_message(_Event.FINISH_SESSION, session_id, {})

    async def on_audio_context_interrupted(self, context_id: str):
        """Cancel the active Volcengine session when the bot is interrupted."""
        await self.stop_all_metrics()
        if (
            context_id
            and context_id in self._active_sessions
            and self._websocket
            and self._websocket.state is State.OPEN
        ):
            try:
                await self._send_session_message(_Event.CANCEL_SESSION, context_id, {})
            except Exception as e:
                logger.warning(f"{self} failed to cancel session {context_id}: {e}")
        self._active_sessions.discard(context_id)
        await super().on_audio_context_interrupted(context_id)

    @property
    def _base_payload(self) -> dict[str, Any]:
        return {"user": {"uid": self._uid}, "namespace": "BidirectionalTTS"}

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Generate speech from text using Volcengine's bidirectional streaming API.

        Args:
            text: The text to synthesize into speech.
            context_id: The context ID, also used as the Volcengine session id.

        Yields:
            None — audio frames arrive asynchronously via the receive task.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")
        try:
            if not self._websocket or self._websocket.state is not State.OPEN:
                await self._connect()

            # Start a new Volcengine session the first time we see this context.
            if context_id not in self._active_sessions:
                await self._send_session_message(
                    _Event.START_SESSION,
                    context_id,
                    dict(self._base_payload, req_params=self._req_params()),
                )
                self._active_sessions.add(context_id)

            await self._send_session_message(
                _Event.TASK_REQUEST,
                context_id,
                dict(self._base_payload, req_params=self._req_params(text)),
            )
            await self.start_tts_usage_metrics(text)
            yield None
        except Exception as e:
            logger.error(f"{self} error in run_tts: {e}")
            yield ErrorFrame(error=f"Volcengine TTS error: {e}")

    async def _receive_messages(self):
        """Receive and dispatch Volcengine binary protocol messages."""
        async for message in self._get_websocket():
            if not isinstance(message, (bytes, bytearray)):
                continue
            try:
                msg = _Message.unmarshal(bytes(message))
            except ValueError as e:
                logger.warning(f"{self} malformed server frame: {e}")
                continue
            await self._handle_server_message(msg)

    async def _handle_server_message(self, msg: _Message):
        """Route a parsed server message to audio contexts and frames."""
        context_id = msg.session_id or self.get_active_audio_context_id()

        if msg.type == _MsgType.AUDIO_ONLY_SERVER:
            if msg.payload:
                await self.stop_ttfb_metrics()
                await self.append_to_audio_context(
                    context_id,
                    TTSAudioRawFrame(
                        audio=msg.payload,
                        sample_rate=self.sample_rate,
                        num_channels=1,
                        context_id=context_id,
                    ),
                )
            return

        if msg.type == _MsgType.ERROR:
            await self._fail_context(context_id, f"server error {msg.error_code}: {msg.payload!r}")
            return

        if msg.event in (_Event.SESSION_FINISHED, _Event.TTS_ENDED):
            await self.stop_ttfb_metrics()
            self._active_sessions.discard(context_id)
            if context_id and self.audio_context_available(context_id):
                await self.append_to_audio_context(
                    context_id, TTSStoppedFrame(context_id=context_id)
                )
                await self.remove_audio_context(context_id)
        elif msg.event == _Event.SESSION_CANCELED:
            # Server acknowledged a CancelSession (interruption). The audio
            # context was already torn down by the interruption handler.
            self._active_sessions.discard(context_id)
        elif msg.event in (_Event.SESSION_FAILED, _Event.CONNECTION_FAILED):
            await self._fail_context(context_id, f"{_Event(msg.event).name}: {msg.payload!r}")
        # SESSION_STARTED / TTS_SENTENCE_START / TTS_SENTENCE_END / TTS_RESPONSE /
        # CONNECTION_STARTED / CONNECTION_FINISHED are progress/metadata — ignored.

    async def _fail_context(self, context_id: str | None, error: str):
        """Report an error and close the affected audio context."""
        logger.error(f"{self} {error}")
        await self.stop_all_metrics()
        self._active_sessions.discard(context_id)
        if context_id and self.audio_context_available(context_id):
            await self.append_to_audio_context(context_id, TTSStoppedFrame(context_id=context_id))
            await self.remove_audio_context(context_id)
        await self.push_error(error_msg=f"Volcengine TTS error: {error}")

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert a Language enum to the service language format.

        Volcengine selects language by ``voice``, so this is a passthrough.

        Args:
            language: The language to convert.

        Returns:
            The language value unchanged.
        """
        return language
