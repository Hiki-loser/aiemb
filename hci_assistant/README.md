# Smart Home Intelligent Assistant (HCI)

Voice-controlled smart home assistant running on Qualcomm QCS6490 (SC171) embedded board, with Vosk streaming Chinese ASR + llama.cpp LLM inference.

## Architecture

```
arecord → Vosk(streaming) → text queue(sliding window) → wake word "小电"
                                                              │
                                                  ┌───────────┘
                                                  ▼
                                           pre-wake ring buffer(3s)
                                                  +
                                           post-wake capture(10s)
                                                  │
                                                  ▼
                                          espeak-ng TTS "收到"(blocking)
                                                  │
                                                  ▼
                                          Vosk re-transcribe(post-wake audio)
                                                  │
                                                  ▼
                                          llama.cpp Qwen2.5 0.5B → parse command
                                                  │
                                                  ▼
                                          Execute action → TTS response
```

## Hardware

| Component | Device | Path |
|-----------|--------|------|
| Speaker (3.5mm) | lahaina-yupikiot | `plughw:0,0`, 48kHz S16_LE |
| Microphone | C-Media USB / K03S | `plughw:2,0` or `plughw:1,0` |
| ASR Model | Vosk Kaldi CN | `/home/fibo/AI_model/vosk_cn/` |
| LLM Model | Qwen2.5 0.5B GGUF | Q4_K_M, 469MB |

## File Structure

```
/home/fibo/project/HCI/
├── main.py                # Entry point
├── config.py              # All paths, constants, device registry
├── pipeline.py            # 4-phase state machine orchestrator
├── wake_word.py           # Vosk streaming + wake word + command capture
├── asr_engine.py          # Vosk model wrapper
├── llm_engine.py          # llama.cpp subprocess manager
├── prompt_templates.py    # LLM prompt (306 chars)
├── action_executor.py     # Parse LLM output → execute actions
├── smart_home_state.py    # Singleton device state store (10 devices)
├── utils.py               # Audio normalization, resample, TTS, logging
├── sounds/
│   ├── wozai.wav          # Wake response "收到"
│   ├── ready.wav          # Startup "智能家居助手已就绪"
│   └── ding.wav           # Alert tone
├── logs/
│   ├── pipeline.log
│   └── errors.log
└── output/                # Debug artifacts
```

## Key Parameters

| Parameter | Value | Config Key |
|-----------|-------|------------|
| Wake word | 小电 (fuzzy: 小点/小店/笑电/小垫) | `WAKE_PATTERN` |
| Wake response | 收到 (blocking playback) | `WAKE_SOUND` |
| Pre-wake buffer | 3 seconds | `PRE_WAKE_SEC` |
| Command capture | Up to 10 seconds | `MAX_POST_WAKE_SEC` |
| Recognizer reset | Every 18 seconds | `RECOGNIZER_RESET_SEC` |
| LLM prompt | 306 chars | `build_prompt()` |
| LLM tokens | 60 max | `LLM_TOKENS` |
| ASR sample rate | 16kHz | `SAMPLE_RATE` |
| Speaker sample rate | 48kHz S16_LE | `SPEAKER_RATE` |

## Smart Home Devices

| Device ID | Type | Label |
|-----------|------|-------|
| living_room_light | light | 客厅灯 |
| bedroom_light | light | 卧室灯 |
| kitchen_light | light | 厨房灯 |
| living_room_ac | ac | 客厅空调 |
| bedroom_ac | ac | 卧室空调 |
| living_room_curtain | curtain | 客厅窗帘 |
| bedroom_curtain | curtain | 卧室窗帘 |
| tv | tv | 电视 |
| air_purifier | purifier | 净化器 |
| water_heater | heater | 热水器 |

## Usage

```bash
# SSH to device
ssh fibo@<device-ip>

# Kill any previous instance
killall python3 pulseaudio 2>/dev/null

# Start (debug mode shows real-time ASR output)
cd /home/fibo/project/HCI
python3 main.py --debug

# Normal mode
python3 main.py
```

## Interaction Flow

1. Say **"小电"** → plays "收到" (blocking)
2. Say your command within 10s, e.g. **"打开客厅灯"**
3. LLM processes → TTS response plays through 3.5mm
4. Returns to listening for next wake word

## Deployment

```bash
# From host (x86_64)
cd /home/hiki/Code/school/aiemb/hci_assistant/
scp *.py fibo@<device-ip>:/home/fibo/project/HCI/
scp sounds/*.wav fibo@<device-ip>:/home/fibo/project/HCI/sounds/
```

## Dependencies (on device)

- Python 3.8+ (system)
- `vosk` (pip): streaming ASR
- `espeak-ng` (apt): TTS
- `numpy`, `wave`, `struct`, `json`, `subprocess` (stdlib)
- llama.cpp binaries (pre-compiled)
- Vosk CN model at `/home/fibo/AI_model/vosk_cn/`

## Audio Setup

Internal 3.5mm jack requires tinymix HPH path configuration on startup (handled automatically by `pipeline._setup_hph()`). Key controls: 89, 90, 106, 107, 111, 112, 133, 134, 136, 137, 242-244, 268, 269, 519.

PulseAudio must be stopped before use (`killall pulseaudio`).

## Wake Word Pattern

```
[小笑][\s]?[电点店垫典颠]
```

Matches: 小电, 小点, 小店, 小 电, 笑电, 小垫, 小典, 小颠
Rejects: (matches anywhere in text for flexibility)
