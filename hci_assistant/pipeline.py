"""Vosk streaming smart home assistant with audio feedback.

Pipeline 流程:
  Phase 1: arecord 持续录音 → Vosk 流式识别 → 检测唤醒词 "小电"
  Phase 2: 回溯 3s 唤醒前音频 + 捕获最长 10s 唤醒后音频
  Phase 3: 播放 "收到" (同步) → Vosk 重新转录 → LLM 解析指令
  Phase 4: 执行动作 + TTS 语音回复
"""

import os
import sys
import re
import time
import logging
import subprocess
import threading

import config
from asr_engine import VoskASR
from llm_engine import LLMEngine
from wake_word import StreamingWakeDetector
from prompt_templates import build_prompt
from action_executor import parse_llm_output, execute
from utils import setup_logging, ts, speak


def play_sound(path, blocking=False):
    """通过 ALSA 播放 WAV 音频文件.
    Args:
        path: WAV 文件路径 (需 48kHz S16_LE mono)
        blocking: True=阻塞等待播放完成, False=后台线程异步播放
    内置 3.5mm 使用 plughw:0,0, 显式指定 S16_LE/48kHz/mono 避免 hw params 错误
    """
    cmd = ["timeout", "6", "aplay", "-D", config.SPEAKER_DEVICE,
           "-f", "S16_LE", "-r", "48000", "-c", "1",
           "--buffer-size=65536", "--period-size=4096", path]
    if blocking:
        subprocess.run(cmd, capture_output=True)
    else:
        threading.Thread(target=lambda: subprocess.run(cmd, capture_output=True), daemon=True).start()


class Pipeline:
    """主控状态机, 持有 ASR/LLM/Wake 三个子系统的引用."""

    def __init__(self, debug=False):
        """初始化所有子系统.
        Args:
            debug: True=实时打印 ASR 部分结果 + 控制台日志
        启动顺序: HPH 音频路径 → 日志 → Vosk 模型 → LLM → 启动音
        """
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(config.LOG_DIR, exist_ok=True)

        self.log = setup_logging(config.LOG_DIR, debug)
        self.log.info("=== Smart Home Assistant starting ===")
        self.debug = debug

        self.log.info("Loading Vosk ASR model...")
        self.asr = VoskASR()          # 加载 ~700MB Vosk CN 模型

        self.log.info("Loading LLM...")
        self.llm = LLMEngine()        # 后续通过 Popen 按需调用 llama.cpp

        self.wake = StreamingWakeDetector(self.asr, debug=debug)

        # 启动音: "智能家居助手已就绪" (同步, 阻塞)
        if os.path.exists(config.STARTUP_SOUND):
            play_sound(config.STARTUP_SOUND, blocking=True)

        self.log.info("Ready.")
        print("\n" + "=" * 50)
        print("Smart Home Assistant")
        print("Wake word: 小电")
        print("=" * 50 + "\n")

    def run(self):
        """主循环: 永远运行, Ctrl+C 退出."""
        while True:
            try:
                # Phase 1+2: 流式录音 → 检测唤醒词 → 捕获指令音频
                command_text = self.wake.listen()

                # Phase 3: 播放 "收到" (同步, 确保用户听完再识别)
                if os.path.exists(config.WAKE_SOUND):
                    play_sound(config.WAKE_SOUND, blocking=True)

                # 去除指令文本中残留的唤醒词前缀
                command_text = re.sub(r'^\s*[小笑][\s]?[电点店垫典颠][,!.，。！？、\s]*', '', command_text).strip()
                command_text = re.sub(r'^\s*start[,!.，。！？、\s]*', '', command_text, flags=re.IGNORECASE).strip()

                self.log.info("Command: %r", command_text)
                print("\n[Command] %s" % (command_text if command_text else "(empty)"))

                if not command_text or len(command_text) < 1:
                    print("[Skip] No command\n")
                    continue

                # Phase 4: LLM 推理 (306 chars prompt)
                print("[LLM] Processing...")
                prompt = build_prompt(command_text)
                llm_output = self.llm.query(prompt)
                self.log.info("LLM: %r", llm_output)

                # Phase 5: 解析 + 执行动作
                parsed = parse_llm_output(llm_output)
                feedback = execute(parsed)
                print("[Assistant] %s" % llm_output.strip()[:120])
                print("[Action] %s\n" % feedback)
                # TTS 播报回复 (非阻塞)
                speak(llm_output.strip().replace("[end of text]", "")[:200])
                self.log.info("Done: %s", feedback)

            except KeyboardInterrupt:
                raise
            except Exception as e:
                self.log.exception("Error: %s", e)
                print("[Error] %s" % e)
                time.sleep(1)

    def shutdown(self):
        """优雅退出: 杀死 LLM 子进程."""
        self.log.info("Shutting down...")
        try: self.llm.shutdown()
        except: pass
        self.log.info("Done")
