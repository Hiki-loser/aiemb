"""Vosk streaming ASR engine. Real-time speech recognition with partial results."""

import os
import sys
import json
import time
import struct
import logging
import subprocess
import threading

import vosk
import config

sys.path.insert(0, config.FIBOAISDK_PATH)
os.environ["RUNTIME_LIB_PATHS"] = config.FIBOAISDK_PATH

_log = logging.getLogger("pipeline")


class VoskASR:
    """Streaming ASR via Vosk Kaldi Chinese model.
    Provides: continuous audio feed, partial results, wake word detection."""

    def __init__(self):
        self._log = _log
        self._log.info("Loading Vosk model: %s", config.VOSK_MODEL_DIR)
        self._model = vosk.Model(config.VOSK_MODEL_DIR)
        self._log.info("Vosk model loaded")

    def create_recognizer(self):
        """Create a new KaldiRecognizer for a streaming session."""
        rec = vosk.KaldiRecognizer(self._model, config.SAMPLE_RATE)
        rec.SetWords(True)
        return rec

    def transcribe_file(self, wav_path, boost=True):
        """Transcribe a WAV file using streaming simulation. Returns final text."""
        import wave
        from utils import peak_amplitude

        if not os.path.exists(wav_path):
            return ""

        with wave.open(wav_path, "rb") as w:
            if w.getframerate() != config.SAMPLE_RATE:
                self._log.warning("Wrong sample rate: %d", w.getframerate())
            audio = w.readframes(w.getnframes())

        # Auto-boost weak signal
        if boost:
            peak = peak_amplitude(wav_path)
            if 0 < peak < 8000:
                gain = min(20000 / peak, 20.0)
                samples = list(struct.unpack(f"<{len(audio)//2}h", audio))
                samples = [int(max(-32768, min(32767, s * gain))) for s in samples]
                audio = struct.pack(f"<{len(samples)}h", *samples)
                self._log.debug("Boosted %.1fx for peak %d", gain, peak)

        rec = self.create_recognizer()
        chunk = int(config.SAMPLE_RATE * 2 * 0.2)  # 200ms

        for i in range(0, len(audio), chunk):
            data = audio[i:i+chunk]
            if len(data) < chunk:
                break
            rec.AcceptWaveform(data)

        final = json.loads(rec.FinalResult())
        return final.get("text", "").strip()
