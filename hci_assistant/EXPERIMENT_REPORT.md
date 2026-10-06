# 智能家居语音助手实验报告

## 一、实验概述

### 1.1 目标

在 Qualcomm QCS6490 (SC171) 嵌入式开发板上构建一个离线语音助手，集成流式语音识别（ASR）、大语言模型（LLM）指令解析、文本转语音（TTS）反馈和智能家居设备模拟控制。

### 1.2 硬件平台

| 项目 | 规格 |
|------|------|
| 芯片 | Qualcomm QCS6490 (Snapdragon 7-series IoT) |
| CPU | 8 核 (4×Cortex-A55 + 4×Cortex-A78, 最高 2.7GHz) |
| RAM | 8GB |
| 存储 | 128GB |
| 系统 | Ubuntu 20.04 (glibc 2.31, Linux 5.4.219) |
| GPU | Adreno 643 (Vulkan 1.1.128) |
| DSP | Hexagon (QNN HTP) |

### 1.3 外设

| 设备 | 型号 | 接口 | 用途 |
|------|------|------|------|
| 麦克风 | C-Media USB PnP | USB, 16kHz S16_LE | 语音录入 |
| 音箱 | 3.5mm 内置接口 | lahaina-yupikiot DSP | 音频输出 |

---

## 二、ASR 语音识别

### 2.1 模型选型实验

在设备上测试了三种 ASR 方案：

| 方案 | 模型 | 大小 | 后端 | 结果 |
|------|------|------|------|------|
| Whisper Tiny | fiboaisdk ONNX | 103MB | CPU | 可用，~0.5s/5s音频，输出繁体中文 |
| Whisper Small | fiboaisdk ONNX | 359MB | CPU | **OOM** — 推理时需 >2GB RAM |
| fiboasr | SNPE 2.29 DSP | 228MB | DSP | **不兼容** — QNN skeleton 版本不匹配 |
| **Vosk Kaldi CN** | vosk 开源 | ~700MB | CPU 流式 | **采用** — 流式识别，~2.7x 实时速度 |

### 2.2 Vosk 流式 ASR

Vosk 是基于 Kaldi 的中文语音识别模型，核心优势：
- **流式识别**：每 100-200ms 输出部分结果（`PartialResult`）
- **端点检测**：内置 VAD（`AcceptWaveform` 返回 True 表示完整语句结束）
- **多候选**：`SetMaxAlternatives(3)` 获取 3 个最佳识别结果
- **原生简体中文**：无需额外繁简转换
- **开源无 License**：不需要 fiboaisdk 授权

模型结构：
```
am/final.mdl (93MB)          — 声学模型
graph/HCLG.fst (559MB)       — 解码图
graph/words.txt (16MB)       — 词汇表
ivector/final.ie (10MB)      — 说话人自适应
rnnlm/final.raw (59MB)       — RNN 语言模型（OOM 时移除）
```

RNNLM 在加载时导致 OOM，移除后模型占用约 700MB，识别准确度可接受。

### 2.3 简体中文输出

Whisper Tiny 原生输出繁体中文，fiboaisdk OpenCC 转换因 SDK 路径硬编码而失效。通过在 Python 层内联 170+ 字符的简繁映射表完成转换。

Vosk 原生输出简体中文，无此问题。

---

## 三、LLM 大语言模型

### 3.1 方案选择

| 方案 | 后端 | 结果 |
|------|------|------|
| QNN DSP (Qwen3 0.6B) | QNN HTP v2.28 | **不兼容** — 设备 QNN 2.29 |
| Vulkan GPU (llama.cpp) | Adreno 643 | **不可用** — Vulkan 1.1 太旧，GPU busy=0% |
| MNN GPU | Adreno 643 | **不可用** — 同上 |
| ONNX Runtime | CPU | 350ms/forward, 可行但无法交互 |
| **llama.cpp CPU** | CPU (8核) | **采用** — 13 tok/s 生成速度 |

### 3.2 llama.cpp 配置

```bash
llama-completion \
  -m qwen2.5-0.5b-instruct-q4_k_m.gguf (469MB Q4_K_M) \
  -f /dev/stdin --single-turn --no-conversation \
  --temp 0.1 -n 60 -t 4 --cpu-mask 0xFF
```

**Prompt 优化过程**：

| 版本 | 长度 | 格式 | 速度 |
|------|------|------|------|
| v1 | 1400+ chars | `DEVICE:x\|ACTION:y\|PARAMS:z` | ~12s |
| v2 | 306 chars | 自然语言回复 | ~5s |

最终 prompt 使用 ChatML 格式，要求 LLM 用一句自然语言回复执行的操作。

### 3.3 glibc 兼容性

交叉编译的 llama.cpp 需要 glibc ≥2.34，设备为 2.31。通过提取 Ubuntu 24.04 ARM64 glibc 库到 `/home/fibo/newlibc/` 解决：

```bash
export LD_LIBRARY_PATH=/home/fibo/newlibc:/home/fibo/miniconda3/lib
/home/fibo/newlibc/ld-linux-aarch64.so.1 ./llama-completion [args]
```

---

## 四、TTS 文本转语音

### 4.1 方案测试

| 方案 | 结果 |
|------|------|
| fibotts (QNN 2.26 DSP) | **不兼容** — 版本不匹配，硬编码 Android 路径 |
| piper | 未安装 |
| **espeak-ng** | **采用** — apt 安装，支持中文 |

### 4.2 espeak-ng 集成

```bash
espeak-ng -v zh -w output.wav "文本内容"
```

生成 22050Hz WAV → Python 重采样到 48000Hz S16_LE → aplay 播放。

---

## 五、音频系统

### 5.1 内置 3.5mm 接口调试

SC171 的 3.5mm 音频接口由 Qualcomm lahaina-yupikiot DSP 驱动，需要通过 tinymix 配置 HPH (Headphone) 音频路径。

**关键发现**：tinymix 在带 `-D 0` 参数时 SET 命令全部失败（"Enclose strings in quotes" 错误），而不带 `-D` 参数时可以正常工作。这是 Qualcomm 定制 tinyalsa 的 bug。

**HPH 路径配置**（控制编号）：
```
89,90: AIF1_PB        — 音频接口路由
106,107: CLSH_DSM_OUT  — DEM 调制器
111,112: RX0,RX1       — 接收通道
133,134: 124 (max)     — 数字音量
136,137: 1             — HPH 开关
242: CLS_H_HIFI        — HPH 模式
243,244: 1             — 补偿开关
268,269: 1             — 额外开关
519: 1                 — 全局使能
```

**播放设备**：`plughw:0,0`（Card 0, MultiMedia1），显式指定 `-f S16_LE -r 48000 -c 1`。

**PulseAudio 冲突**：PulseAudio 持续占用音频设备导致 aplay 失败，需 `killall pulseaudio`。PulseAudio 会被 dbus 自动重启，但重启后的实例处于 D 状态（不可中断睡眠），不影响直接 aplay。

### 5.2 外置 USB 设备

| 设备 | Card | 格式 | 备注 |
|------|------|------|------|
| C-Media USB Mic | Card 2 | 16kHz S16_LE | 需衰减 gain=0.3 (削波) |


### 5.3 多 Card 布局变化

设备每次重启或 USB 热插拔后，ALSA Card 编号会发生变化。配置文件使用 `MIC_CONFIGS` 字典方式进行抽象，用户只需修改 `ACTIVE_MIC` 编号即可切换麦克风。

---

## 六、唤醒词检测

### 6.1 方案设计

采用流式文本队列 + 正则匹配方案：

```
arecord → Vosk partial → deque(maxlen=40) → regex search
```

### 6.2 唤醒词演变

| 版本 | 唤醒词 | 模式 |
|------|--------|------|
| v1 | start | `\bstart\b` (英文) |
| v2 | 开始 | `(?:^\|[\s,...])开始` (中文左边界) |
| v3 | **小电** | `[小笑][\s]?[电点店垫典颠]` (模糊匹配) |

### 6.3 模糊匹配

正则 `[小笑][\s]?[电点店垫典颠]` 可匹配 Vosk 可能的识别变体：
- 小电, 小点, 小店, 笑电, 小垫, 小典, 小颠
- 小 电 (带空格)
- 任意位置匹配（无边界限制）

### 6.4 缓冲区管理

| 缓冲区 | 大小 | 策略 |
|--------|------|------|
| `text_queue` | deque(maxlen=40) ~4s | FIFO 自动出队 |
| `audio_ring` | 3 秒 | 滑动窗口，唤醒时一起捕获 |
| `post_wake_buf` | 10 秒 | 唤醒后收集指令 |

### 6.5 Recognizer 重置

Vosk decoder lattice 在连续识别时无限累积，18 秒无唤醒词后自动重置 recognizer 并清空文本/音频缓冲区，防止内存耗尽。

---

## 七、Pipeline 流程

### 7.1 启动序列

1. 配置 3.5mm HPH 音频路径 (tinymix × 16 控制)
2. 加载 Vosk CN 模型（~3s, ~700MB）
3. 启动 llama.cpp 子进程
4. **播放 "智能家居助手已就绪"**（同步，阻塞）
5. 进入主循环

### 7.2 主循环

```
Phase 1: 流式录音 → Vosk 部分结果 → 检测 "小电"
         │
         ├─ 文本已在队列中(≥4字) → 直接作为指令
         └─ 文本为空/碎片 → 进入 Phase 2
              │
Phase 2: 回溯 3s 音频(含"小电") + 继续录音最长 10s
         │
Phase 3: 播放 "收到"（同步，阻塞等待播放完成）
         │
Phase 4: Vosk 重新转录 post-wake 音频 (MaxAlternatives=3)
         │
Phase 5: LLM 推理 (306 chars prompt, ~3-5s)
         │
Phase 6: 执行动作 + TTS 语音回复
         │
         └→ 回到 Phase 1
```

### 7.3 性能数据

| 阶段 | 耗时 |
|------|------|
| Vosk 模型加载 | ~3s |
| LLM 模型加载 | ~1.5s (冷启动) |
| 唤醒词检测 | ~0.3s/cycle (含 0.1s 录音处理) |
| 指令音频捕获 | 最长 10s |
| 指令转写 (Vosk) | ~0.5s |
| LLM 推理 | ~3-5s |
| TTS 生成+播放 | ~1s |

---

## 八、遇到的重大问题及解决方案

### 8.1 QNN SDK Converter Bug

**问题**：QNN 2.29.0 ONNX converter 在 Cast→Unsqueeze/Reshape 链路中 C 层形状推导失败。

**尝试**：删除 104 个 float32→float32 Cast，替换为 Identity；Unsqueeze→显式 Reshape；raw_data 编码 shape tensor；--input_dim 显式设定。**均失败**。

**结论**：QNN SDK 2.29.0 存在无法绕过的 C 层 shape inference bug，无法将 ONNX 模型转换为 DLC 格式。

### 8.2 GPU 不可用

MNN Vulkan 和 llama.cpp Vulkan 后端均无法使用 Adreno 643 GPU：
- `gpubusy` 始终为 0/0（GPU 空闲）
- MNN Vulkan 推理速度与 CPU 完全相同（545ms）
- 根本原因：Adreno 643 Vulkan 1.1.128 < 1.2 最低要求

### 8.3 3.5mm 音频输出

历时最长的问题。tinymix `-D 0` 参数导致所有 SET 命令失败。发现不带 `-D` 标志的 tinymix 可以正常设置控制。同时发现控制 519 必须在 HPH 路径配置中设置，否则设备 0 拒绝硬件参数。

### 8.4 K03S 录音静音

K03S 耳机在 `S16_LE` 格式下录音为全零。必须使用原生 `S24_3LE` 格式，在软件层完成 24bit→16bit 转换。

### 8.5 Vosk Decoder Lattice OOM

连续识别新闻语音时，Vosk decoder lattice 无限增长导致内存耗尽。通过每 18 秒重置 recognizer 解决，同时清空文本和音频缓冲区。

### 8.6 唤醒后指令丢失

唤醒词 "小电" 所在音频块未被包含在 post-wake 缓冲区中，导致紧随其后的指令丢失。修复：唤醒时回溯 3 秒 pre-wake 音频并一起送入转录。

---

## 九、智能家居设备模拟

### 9.1 设备注册表

| 设备 ID | 类型 | 标签 | 操作 |
|---------|------|------|------|
| living_room_light | light | 客厅灯 | turn_on/off, set_brightness |
| bedroom_light | light | 卧室灯 | turn_on/off, set_brightness |
| kitchen_light | light | 厨房灯 | turn_on/off, set_brightness |
| living_room_ac | ac | 客厅空调 | turn_on/off, set_temperature, set_mode |
| bedroom_ac | ac | 卧室空调 | turn_on/off, set_temperature, set_mode |
| living_room_curtain | curtain | 客厅窗帘 | open, close |
| bedroom_curtain | curtain | 卧室窗帘 | open, close |
| tv | tv | 电视 | turn_on/off, set_channel, set_volume |
| air_purifier | purifier | 净化器 | turn_on/off, set_mode |
| water_heater | heater | 热水器 | turn_on/off, set_temperature |

### 9.2 指令解析

LLM 输出自然语言回复（如 "客厅灯已打开，亮度80%"），`action_executor.py` 通过关键词匹配提取设备、动作和参数：

```
打开/开启 → turn_on
关闭/关掉 → turn_off
N度 → temperature=N
亮度N → brightness=N
```

---

## 十、文件清单

### 10.1 设备端 `/home/fibo/project/HCI/`

| 文件 | 行数 | 职责 |
|------|------|------|
| main.py | 44 | 入口，信号处理 |
| config.py | 68 | 全部路径/常量/设备注册表 |
| pipeline.py | 127 | 主控状态机，HPH 音频初始化 |
| wake_word.py | 228 | Vosk 流式，唤醒词，指令捕获 |
| asr_engine.py | 90 | Vosk 模型封装 |
| llm_engine.py | 90 | llama.cpp 子进程管理 |
| prompt_templates.py | 31 | LLM prompt (306 字符) |
| action_executor.py | 115 | 指令解析+动作执行 |
| smart_home_state.py | 130 | 设备状态单例 |
| utils.py | 250 | 音频重采样/T2S/TTS/日志 |

### 10.2 音频资源

| 文件 | 内容 | 时长 | 格式 |
|------|------|------|------|
| sounds/ready.wav | "智能家居助手已就绪" | 2.6s | 48kHz S16_LE |
| sounds/wozai.wav | "收到" | 0.8s | 48kHz S16_LE |
| sounds/ding.wav | "叮" | 0.5s | 48kHz S16_LE |

### 10.3 AI 模型

| 模型 | 路径 | 大小 |
|------|------|------|
| Vosk CN | `/home/fibo/AI_model/vosk_cn/` | ~700MB |
| Qwen2.5 0.5B | `/home/fibo/llama-vulkan-aarch64-musl/qwen2.5-0.5b-instruct-q4_k_m.gguf` | 469MB |

---

## 十一、总结

本项目在 SC171 嵌入式平台上实现了一个完整的离线语音助手，核心指标：

- **ASR**: Vosk 流式中文本地识别，~2.7x 实时速度，原生简体中文
- **LLM**: llama.cpp Qwen2.5 0.5B Q4_K_M，~13 tok/s 生成，306 字符 prompt，~3-5s 响应
- **TTS**: espeak-ng 中文语音合成，通过 3.5mm 接口输出
- **唤醒**: "小电" 模糊匹配，支持同音字和拼写变体
- **Pipeline**: 滑动窗口文本检测 + 音频回溯 + 端点检测，全流程 ~5-8s
- **可靠性**: Recognizer 定时重置、缓冲区上限管理、PulseAudio 冲突处理
