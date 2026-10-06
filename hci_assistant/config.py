"""Central configuration for Smart Home HCI Assistant."""

# === llama.cpp LLM ===
LLAMA_LD = "/home/fibo/newlibc/ld-linux-aarch64.so.1"
LLAMA_BIN = "/home/fibo/llama-b9500-new/llama-completion"
LLAMA_MODEL = "/home/fibo/llama-vulkan-aarch64-musl/qwen2.5-0.5b-instruct-q4_k_m.gguf"
LLAMA_WORKDIR = "/home/fibo/llama-cross2"
LLAMA_LD_PATH = "/home/fibo/newlibc:/home/fibo/miniconda3/lib:/home/fibo/llama-cross2"

LLM_TEMP = 0.1
LLM_TOKENS = 60
LLM_THREADS = 4
LLM_CPU_MASK = "0xFF"          # All 8 cores (ASR runs between LLM calls)
LLM_PROMPT_CACHE = "/tmp/llm_cache.bin"
STOP_MARKER = "<<DONE>>"

# === Vosk ASR (streaming Chinese) ===
VOSK_MODEL_DIR = "/home/fibo/AI_model/vosk_cn/"

# === fiboaisdk (for license + fallback ASR) ===
FIBOAISDK_PATH = "/usr/local/lib/python3.8/dist-packages/fiboaisdk"
LICENSE_DIR = "/home/fibo/qcom_6490_license/"

# === Audio: Microphone configuration ===
# Switch mics by changing ACTIVE_MIC: 1 = USB desk mic, 2 = XIBERIA K03S headset
ACTIVE_MIC = 1

# Card: Card0=C-Media USB Mic, Card1=3.5mm lahaina
ACTIVE_MIC = 1

MIC_CONFIGS = {
    1: {
        "name": "USB Desk Mic (C-Media)",
        "device": "plughw:0,0", "sample_rate": 16000, "channels": 1,
        "format": "S16_LE", "gain": 0.3, "needs_resample": False,
    },
    2: {
        "name": "XIBERIA K03S Headset",
        "device": "", "sample_rate": 48000, "channels": 1,
        "format": "S24_3LE", "gain": 1.0, "needs_resample": True,
        "resample_target": 16000,
    },
}

# Speaker: internal 3.5mm jack = Card 1, device 5 (AFE-PROXY RX)
SPEAKER_DEVICE = "plughw:1,5"
SPEAKER_RATE = 48000

# Pre-recorded audio responses
WAKE_SOUND = "/home/fibo/project/HCI/sounds/wozai.wav"
STARTUP_SOUND = "/home/fibo/project/HCI/sounds/ready.wav"

# Convenience accessors (uses ACTIVE_MIC)
def _mic():
    return MIC_CONFIGS[ACTIVE_MIC]
AUDIO_DEVICE = _mic()["device"]       # Resolved at import time; rebuild after changing ACTIVE_MIC
SAMPLE_RATE = _mic()["sample_rate"]
AUDIO_GAIN = _mic()["gain"]

# === Wake word ===
WAKE_CHUNK_DURATION = 4.0     # seconds per listening chunk (long enough for wake+cmd)
# Wake word "小电" fuzzy match. Homophones: 小电/小点/小店/小垫/笑电/小典/小颠
# Allows optional space between chars, matches anywhere in text
WAKE_PATTERN = r"[小笑][\s]?[电点店垫典颠]"
SILENCE_THRESHOLD = 150       # RMS below this = silence

# === Command recording ===
CMD_MAX_DURATION = 4.0        # max recording seconds for follow-up
CMD_SILENCE_TIMEOUT = 2.0     # stop early after this many silent seconds

# === Output ===
OUTPUT_DIR = "/home/fibo/project/HCI/output/"
LOG_DIR = "/home/fibo/project/HCI/logs/"
