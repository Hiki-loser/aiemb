"""Vosk streaming wake word + command detector with audio capture.

架构:
  arecord 持续输出 PCM → Vosk 每 100ms 产出部分结果 → deque 滑动窗口
  检测到唤醒词 → 回溯 pre-wake 音频 (含 "小电") → 继续录音最长 10s
  → Vosk 重新转录 → 返回指令文本

缓冲区设计:
  text_queue:   deque(maxlen=40) 文本滑动窗口, ~4s, FIFO 自动清理
  audio_ring:   deque, 3s 唤醒前音频, 唤醒时一起捕获
  post_wake_buf: list, 唤醒后音频, 最长 10s (包含回溯的 3s)
"""

import re
import sys
import json
import time
import struct
import logging
import subprocess
from collections import deque

import config


class StreamingWakeDetector:
    """流式唤醒词检测器."""

    def __init__(self, vosk_asr, debug=False):
        """Args:
            vosk_asr: VoskASR 实例 (已加载模型)
            debug: True=实时打印 ASR 部分结果
        """
        self._asr = vosk_asr
        self._pattern = re.compile(config.WAKE_PATTERN, re.IGNORECASE)
        self._log = logging.getLogger("pipeline")
        self._debug = debug
        self._mic = config.MIC_CONFIGS[config.ACTIVE_MIC]

    def listen(self, timeout=None):
        """阻塞等待唤醒词, 返回指令文本.
        唤醒词模糊匹配: 小电/小点/小店/笑电/小垫/小 电 等.
        返回: 指令文本 (可能为空字符串).
        """
        mic = self._mic
        needs_resample = mic.get("needs_resample", False)
        src_rate = mic["sample_rate"]
        ratio = int(src_rate / config.SAMPLE_RATE)  # 48k→16k 降采样比例
        tgt_rate = config.SAMPLE_RATE

        # 根据麦克风类型选用 arecord 参数
        # K03S: S24_3LE 48kHz, plughw 自动转换后软件降采样
        # C-Media: S16_LE 16kHz, 直通
        if needs_resample:
            arecord_cmd = [
                "arecord", "-D", mic["device"], "-f", mic["format"],
                "-r", str(src_rate), "-c", str(mic["channels"]),
                "--buffer-size=2048", "--period-size=1024", "-",
            ]
        else:
            arecord_cmd = [
                "arecord", "-D", mic["device"], "-f", mic["format"],
                "-r", str(src_rate), "-c", str(mic["channels"]),
                "--buffer-size=1024", "--period-size=512", "-",
            ]

        self._log.info("Streaming from %s", mic["name"])
        print("[Streaming] Listening for wake word...")
        sys.stdout.flush()

        proc = subprocess.Popen(arecord_cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, bufsize=0)

        start_time = time.time()
        chunk_in = 6400                # 每次读取的字节数
        RECOGNIZER_RESET_SEC = 18      # recognizer 重置周期 (防止 lattice OOM)
        QUEUE_MAX = 40                 # 文本队列最大长度 (~4s)

        # === 文本滑动窗口 ===
        text_queue = deque(maxlen=QUEUE_MAX)

        # === 唤醒前音频环形缓冲 (3s) ===
        PRE_WAKE_SEC = 3
        audio_ring = deque()
        audio_ring_bytes = 0
        MAX_RING_BYTES = tgt_rate * 2 * PRE_WAKE_SEC

        # === 唤醒后音频捕获 (最长 10s, 含回溯的 3s) ===
        post_wake_buf = []
        post_wake_samples = 0
        MAX_POST_WAKE_BUF = tgt_rate * 2 * 12   # 12s 上限
        MAX_POST_WAKE_SEC = 10.0                 # 目标捕获时长

        wake_detected = False

        def _new_recognizer():
            """创建新的 Vosk recognizer, 开启词级结果."""
            r = self._asr.create_recognizer()
            r.SetWords(True)
            return r

        rec = _new_recognizer()
        rec_started = time.time()

        try:
            while True:
                if timeout and (time.time() - start_time) > timeout:
                    return ""

                data = proc.stdout.read(chunk_in)
                if not data:
                    time.sleep(0.01)
                    continue

                # ---- 48k→16k 降采样 (K03S 需要, C-Media 直通) ----
                if needs_resample:
                    if mic["format"] == "S24_3LE":
                        # 24-bit (3 bytes/sample) → int16 → 3x decimate
                        n = len(data) // 3
                        s16 = [0] * n
                        for i in range(n):
                            b = data[i*3:(i+1)*3]
                            val = b[0] | (b[1] << 8) | (b[2] << 16)
                            if val & 0x800000:
                                val -= 0x1000000  # 符号扩展
                            s16[i] = val >> 8     # 24→16 bit
                        decimated = s16[::ratio]
                        data_16k = struct.pack(f"<{len(decimated)}h", *decimated)
                    else:
                        samples = struct.unpack(f"<{len(data)//2}h", data)
                        decimated = samples[::ratio]
                        data_16k = struct.pack(f"<{len(decimated)}h", *decimated)
                    if len(data_16k) < 100:
                        continue
                else:
                    data_16k = data

                # ---- 唤醒前: 维护音频环形缓冲 (用于回溯) ----
                if not wake_detected:
                    audio_ring.append(data_16k)
                    audio_ring_bytes += len(data_16k)
                    while audio_ring_bytes > MAX_RING_BYTES and len(audio_ring) > 1:
                        audio_ring_bytes -= len(audio_ring.popleft())

                # ---- Recognizer 定期重置: 防止 lattice 内存膨胀 ----
                if not wake_detected and (time.time() - rec_started) > RECOGNIZER_RESET_SEC:
                    rec = _new_recognizer()
                    rec_started = time.time()
                    text_queue.clear()
                    audio_ring.clear()
                    audio_ring_bytes = 0
                    if self._debug:
                        sys.stdout.write("\n[ASR] (reset)\n")
                        sys.stdout.flush()

                # ---- 喂入 Vosk, 获取部分结果 ----
                is_final = rec.AcceptWaveform(data_16k)
                partial = json.loads(rec.PartialResult()).get("partial", "")

                # =================================================================
                # Phase 1: 等待唤醒词
                # =================================================================
                if not wake_detected:
                    if partial:
                        text_queue.append(partial)

                        if self._debug:
                            sys.stdout.write("\r[ASR] %s   " % partial[:80])
                            sys.stdout.flush()

                        # 扫描整个队列的拼接文本
                        combined = " ".join(text_queue)
                        match = self._pattern.search(combined)
                        if match:
                            wake_detected = True
                            print("\n[WAKE] %s" % combined[-120:])
                            sys.stdout.flush()

                            # 尝试从队列文本中提取指令 (用户一口气说完的情况)
                            after_wake = combined[match.end():].strip()
                            after_wake = re.sub(r'^[,!.，。！？、\s]+', '', after_wake)

                            if len(after_wake) >= 4:
                                # 队列已有完整指令, 直接返回 (快速路径)
                                self._log.info("Command from queue: %r", after_wake)
                                return after_wake.strip()

                            # 否则: 回溯 pre-wake 音频 + 当前块, 继续捕获
                            for chunk in audio_ring:
                                post_wake_buf.append(chunk)
                                post_wake_samples += len(chunk)
                            post_wake_buf.append(data_16k)
                            post_wake_samples += len(data_16k)

                # =================================================================
                # Phase 2: 唤醒后捕获指令音频
                # =================================================================
                else:
                    post_wake_buf.append(data_16k)
                    post_wake_samples += len(data_16k)

                    # 上限保护: 丢弃最早的音频块
                    while post_wake_samples > MAX_POST_WAKE_BUF and len(post_wake_buf) > 1:
                        dropped = post_wake_buf.pop(0)
                        post_wake_samples -= len(dropped)

                    post_wake_sec = post_wake_samples / (tgt_rate * 2)

                    # 达到目标时长
                    if post_wake_sec >= MAX_POST_WAKE_SEC:
                        break

                    # Vosk 检测到完整语句 (endpoint)
                    if is_final and partial:
                        break

        except KeyboardInterrupt:
            raise
        finally:
            try: proc.terminate()
            except: pass
            try: proc.wait(timeout=1)
            except: pass

        if not wake_detected:
            return ""

        # ---- 重新转录 post-wake 音频 (干净 recognizer + MaxAlternatives) ----
        post_wake_audio = b"".join(post_wake_buf)
        command = self._accurate_transcribe(post_wake_audio)
        command = self._strip_wake(command)
        self._log.info("Command: %r", command)
        return command.strip()

    def _accurate_transcribe(self, audio_bytes):
        """用新 recognizer + MaxAlternatives=3 重新转录音频, 取最佳结果."""
        if not audio_bytes or len(audio_bytes) < 1600:
            return ""

        rec = self._asr.create_recognizer()
        rec.SetWords(True)
        rec.SetMaxAlternatives(3)  # 获取 3 个候选, 选置信度最高者

        chunk = 6400
        for i in range(0, len(audio_bytes), chunk):
            data = audio_bytes[i:i+chunk]
            if len(data) < 100:
                break
            rec.AcceptWaveform(data)

        final = json.loads(rec.FinalResult())
        text = final.get("text", "").strip()
        alts = final.get("alternatives", [])

        if alts:
            best = max(alts, key=lambda x: x.get("confidence", 0))
            alt_text = best.get("text", "")
            if len(alt_text) > len(text):
                text = alt_text

        return text

    def _strip_wake(self, text):
        """去除指令文本中的唤醒词前缀."""
        text = re.sub(r'^\s*[小笑][\s]?[电点店垫典颠][,!.，。！？、\s]*', '', text)
        text = re.sub(r'^\s*start[,!.，。！？、\s]*', '', text, flags=re.IGNORECASE)
        return text.strip()
