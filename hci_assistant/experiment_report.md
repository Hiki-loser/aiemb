# 智能家居语音助手（HCI）实验报告

## 设备信息

- **开发板**: GuangheTong SC171（广和通）
- **SoC**: Qualcomm QCS6490（IoT 版本 Snapdragon 7-series）
- **CPU**: 8 核（4×Cortex-A55 + 4×Cortex-A78），最高 2.7GHz
- **GPU**: Adreno 643，Vulkan 1.1.128
- **DSP**: Qualcomm Hexagon（QNN HTP）
- **RAM**: 8GB
- **存储**: 128GB
- **OS**: Ubuntu 20.04（GNU/Linux 5.4.219 aarch64）
- **AI SDK**: fiboaisdk v2.29.0（含 QNN/SNPE/MNN/ONNX Runtime/OpenCV/Sherpa-ONNX）

---

## 第一部分：已完成的工作

### 1. GPU 推理可行性探索（2026-06-04）

#### 1.1 llama.cpp Vulkan 尝试

最初的目标是将 Qwen 0.5B 大语言模型（LLM）推理卸载到 Adreno 643 GPU 上，释放 CPU 以运行其他模型。

**环境搭建**：
- 通过 WireGuard VPN 隧道（temp6.conf）连接设备：`ssh fibo@10.0.0.102`
- 在宿主机上交叉编译 llama.cpp（aarch64-linux-gnu-gcc）
- 解决 glibc 版本不匹配问题：设备 glibc 2.31，交叉编译需要 2.34+。从 Ubuntu 24.04 ARM64 包中提取 libc 部署到 `/home/fibo/newlibc/`
- 成功运行：`LD_LIBRARY_PATH=/home/fibo/newlibc:/home/fibo/miniconda3/lib /home/fibo/newlibc/ld-linux-aarch64.so.1 ./llama-cli`

**测试结果**：
- CPU 推理：Qwen2.5 0.5B Q4_K_M GGUF（469MB），生成速度 ~12.9 tok/s，提示处理 ~35 tok/s
- Vulkan 尝试失败：`VK_ICD_FILENAMES=/usr/lib/libvulkan_adreno.so` 可以加载 ICD，但 `vkCreateDevice` 时 SIGSEGV（空指针崩溃）

**根因**：Adreno 643 GPU 的 Vulkan 驱动版本 1.1.128，但 ggml-vulkan 需要 Vulkan ≥1.2。子组大小 64（需要 32），共享内存 32KB（需要 ≥64KB）。

#### 1.2 GPU Busy 监控验证

为了确认 GPU 是否被使用，我们监控了 Qualcomm GPU 工作状态：

```bash
cat /sys/class/kgsl/kgsl-3d0/gpubusy   # GPU 忙碌计数器
cat /sys/class/kgsl/kgsl-3d0/gpuclk    # GPU 时钟频率
```

监控结果：
- llama.cpp 运行时 GPU busy 始终为 0/0
- MNN Vulkan 推理 50 次，GPU busy 仍然为 0/0
- GPU 时钟保持在 315MHz 空闲频率

**结论**：MNN Vulkan 后端静默回退到 CPU 执行，Adreno 643 GPU 无法用于 LLM 推理。

#### 1.3 MNN Vulkan 尝试

MNN（阿里巴巴推理框架）v3.0.5 支持 Vulkan 后端。我们将 ONNX 模型转换为 `.mnn` 格式：

```python
# 在设备上直接转换
import MNN.tools.mnnconvert as converter
converter.main(['-f', 'ONNX', '--modelFile', 'model.onnx', '--MNNModel', 'qwen.mnn'])
```

转换成功（2.4GB 权重），但 MNN Vulkan 创建会话后推理速度和 CPU 完全一致（~545ms/forward），GPU busy 为 0%。**确认 Adreno 643 无法通过 Vulkan 加速 LLM 推理。**

#### 1.4 QNN（Qualcomm Neural Network）DSP 尝试

QNN 是 Qualcomm 的官方推理 SDK，支持 DSP（Hexagon）后端。设备上已有 Qwen3 0.6B 的 DSP 编译模型（`.fmodel` 格式，QNN 2.28 DSP）：

```
/home/fibo/AI_model/llm_models/Qwen3-0.6B/qwen3-0.6b_1.0.0_qcom_6490_qnn_2.28_dsp.fmodel
```

**首次尝试加载失败**：模型内嵌硬编码路径 `/model/models/qwen3_0.6b_qnn_6490/...`，需要 QNN 2.28 的 HTP 库文件。设备上的 fiboaisdk 是 QNN 2.29 版本。

**版本不匹配问题**：
- `.fmodel` 内嵌的 Cap'n Proto 序列化数据包含硬编码的运行时库路径
- 尝试用相同长度的路径替换（`/home/fibo/XX/qwen3_0.6b_qnn_6490/` = 34 字符 = `/model/models/qwen3_0.6b_qnn_6490/`）
- 路径长度匹配但仍然损坏了 Cap'n Proto 数据结构（"expected array.size() >= offset + segmentSize"）
- 使用 `sudo ln -sf` 创建 `/model` 符号链接后，库加载成功但 DSP skeleton 版本不匹配（error 1008：`QNN_COMMON_ERROR_INCOMPATIBLE_BINARIES`）

**最终结论**：Qwen3 0.6B DSP 模型（QNN 2.28）无法在 QNN 2.29 运行时上运行。需要用 QNN 2.29 SDK 重新编译模型。

#### 1.5 QNN SDK ONNX→DLC 转换尝试

在宿主机上使用 QNN SDK 2.29.0（`qnn_sdk/`）将 ONNX 模型转换为 QNN DLC 格式。

**转换过程中遇到的 Bug**：
1. Flatten/Reshape 操作形状推导失败：`calculateShape: Unable to calculate ReshapeOp output shape for op /m/model/Flatten`
2. Cast→Unsqueeze 操作链的形状信息丢失：QNN 在"转换时解释" Cast 操作，但解释后的中间结果形状无法传播到后续的 Unsqueeze/Reshape
3. 尝试了 6 种修复方法均失败：
   - 删除 float32→float32 Cast（104 个）
   - 替换 Cast 为 Identity 通过操作
   - 替换 Unsqueeze→Reshape 使用显式形状
   - 使用 raw_data 编码形状张量
   - 使用 `--input_dim` 显式指定输入形状
   - 使用 `--no_simplification` 跳过 ONNX 简化

**根因**：QNN SDK 2.29.0 的 C 层形状推导引擎存在 bug，无法正确处理 Transformer 模型中的 Cast→Unsqueeze/Reshape 操作链。

#### 1.6 GPU 推理尝试总结

| 方案 | GPU 使用 | 结果 |
|------|---------|------|
| llama.cpp Vulkan | Vulkan 1.1 太旧 | SIGSEGV 崩溃 |
| MNN Vulkan | 0% GPU busy | 静默回退 CPU |
| QNN DSP（Qwen3 0.6B） | 2.28→2.29 版本不匹配 | DSP skeleton 加载失败 |
| QNN SDK Converter | C 层形状推导 bug | 6 种修复均失败 |
| ONNX Runtime CPU | 无 GPU | 350ms/forward（可用） |

**最终结论：Adreno 643 GPU 无法在当前硬件和软件栈上用于 LLM 推理。**

### 2. 语音识别（ASR）模型探索

#### 2.1 fiboaisdk Whisper 模型

设备上预装的 ASR 模型：
- **Whisper Tiny**（103MB ONNX CPU）：~0.5s 处理 5s 音频
- **Whisper Small**（359MB ONNX CPU）：加载 OOM（需要 >2GB 推理内存）
- **fiboasr_base**（228MB SNPE/QNN DSP）：模型编译为 QNN 2.28 DSP，与设备 QNN 2.29 不兼容

**中文繁简转换问题**：

Whisper 模型原生输出繁体中文。fiboaisdk 内置 OpenCC 繁简转换，但需要 `.ocd2` 词典文件。解决方案：

```bash
# 将 fiboasr 的 OpenCC 词典复制到 Whisper Tiny 目录
cp /home/fibo/AI_model/asr_models/fiboasr/TSCharacters.ocd2 /home/fibo/AI_model/asr_models/whisper_tiny/
cp /home/fibo/AI_model/asr_models/fiboasr/TSPhrases.ocd2 /home/fibo/AI_model/asr_models/whisper_tiny/
```

在 `Init()` 时传入 T2S 配置：`{"simple_chinese_path": "/path/to/t2s.json"}`。

**Python 内联 T2S 作为后备方案**：当 SDK OpenCC 不工作时，使用 170+ 字符的映射表进行繁→简转换：

```python
T2S_MAP = str.maketrans({
    "國":"国", "際":"际", "會":"会", "動":"动", "為":"为",
    "對":"对", "機":"机", "體":"体", "學":"学", "發":"发",
    # ... 170+ entries
})
def t2s_convert(text):
    return text.translate(T2S_MAP)
```

### 3. Vosk 流式中文 ASR 模型

#### 3.1 模型部署

用户下载了 Vosk 中文模型（`model-cn.zip`，2.1GB），解压后部署到设备 `/home/fibo/AI_model/vosk_cn/`。

模型结构（Kaldi 格式）：
- `am/final.mdl`（93MB）：声学模型（TDNN）
- `graph/HCLG.fst`（559MB）：解码图（WFST）
- `graph/words.txt`（16MB）：词表
- `rescore/G.carpa`（1.2GB）：重打分语言模型（删除以节省内存）
- `rnnlm/`（59MB）：RNN 语言模型（删除以防止 OOM）
- `ivector/`（11MB）：说话人自适应

去掉 rescore（1.2GB）和 rnnlm（59MB）后，模型核心大小约 700MB，可在 8GB 设备上运行。

#### 3.2 Vosk 流式识别验证

```python
import vosk, json, wave

model = vosk.Model("/home/fibo/AI_model/vosk_cn")
rec = vosk.KaldiRecognizer(model, 16000)
rec.SetWords(True)

# 流式喂入音频，每 200ms 获取部分结果
with wave.open("audio.wav", "rb") as w:
    audio = w.readframes(w.getnframes())

chunk = 6400  # 200ms at 16kHz
for i in range(0, len(audio), chunk):
    rec.AcceptWaveform(audio[i:i+chunk])
    partial = json.loads(rec.PartialResult())
    print(partial.get("partial", ""))
```

**测试结果**（央视新闻音频通过 K03S 耳机输入）：
- 实时识别速度：**2.7x 实时**（1.8s 处理 5s 音频）
- 部分结果每 200ms 更新：`"我" → "我 看" → "我 看看" → "我 看 他 和 我 之间 未来"`
- 最终结果：`"我 看 他 和 我 之间 未来 谈谈 的 一点 的 开始"`
- 原生简体中文输出，无需 T2S 转换
- 所有中文识别结果均为简体（与 Whisper 输出繁体不同）

**Vosk vs Whisper Tiny 对比**：

| 特性 | Whisper Tiny | Vosk Kaldi CN |
|------|-------------|---------------|
| 识别方式 | 文件批量 | **实时流式** |
| 部分结果 | 无 | **每 200ms 更新** |
| 中文精度 | 一般 | **更好** |
| 简繁 | 繁体需转换 | **原生简体** |
| 速度 | ~1s/5s | **1.8s/5s（2.7x 实时）** |
| 依赖 | fiboaisdk + license | 仅 vosk pip 包 |
| 内存 | ~103MB | ~700MB |
| 流式 API | 无 | `AcceptWaveform()` + `PartialResult()` |

### 4. espeak-ng TTS 文本转语音

#### 4.1 安装与测试

```bash
sudo apt install espeak-ng
```

fiboaisdk TTS 模型（`fibotts_1.0.0_qcom_6490-8550_qnn_2.26_dsp`）需要 QNN 2.26 DSP，与设备 QNN 2.29 不兼容。选择 espeak-ng 作为替代方案。

#### 4.2 WAV 生成

```python
import subprocess, wave, struct

def speak(text, lang="zh"):
    raw = "/tmp/tts_raw.wav"
    out = "/tmp/tts.wav"
    subprocess.run(["espeak-ng", "-v", lang, "-w", raw, text], capture_output=True)

    # espeak-ng 生成 22050Hz → 重采样到 48000Hz（音频硬件要求）
    with wave.open(raw, "rb") as w:
        p = w.getparams()
        d = w.readframes(w.getnframes())
    samples = struct.unpack(f"<{len(d)//2}h", d)
    ratio = p.framerate / 48000
    new = [samples[int(i * ratio)] for i in range(int(len(samples) / ratio))]
    with wave.open(out, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(48000)
        w.writeframes(struct.pack(f"<{len(new)}h", *new))
    return out
```

**预录制音频**：
- `wozai.wav`："收到"（0.8s, 48kHz）
- `ready.wav`："智能家居助手已就绪"（2.6s, 48kHz）

### 5. llama.cpp LLM 推理引擎

#### 5.1 部署配置

- **二进制**: `/home/fibo/llama-b9500-new/llama-completion`
- **模型**: `/home/fibo/llama-vulkan-aarch64-musl/qwen2.5-0.5b-instruct-q4_k_m.gguf`（469MB, Q4_K_M 量化）
- **glibc 解决方案**: `/home/fibo/newlibc/ld-linux-aarch64.so.1`
- **库路径**: `LD_LIBRARY_PATH=/home/fibo/newlibc:/home/fibo/miniconda3/lib:/home/fibo/llama-cross2`

#### 5.2 调用命令

```bash
/home/fibo/newlibc/ld-linux-aarch64.so.1 \
  /home/fibo/llama-b9500-new/llama-completion \
  -m /home/fibo/llama-vulkan-aarch64-musl/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  -f /dev/stdin --temp 0.1 -n 60 -t 4 --cpu-mask 0xFF \
  --no-conversation --no-display-prompt --single-turn
```

#### 5.3 Prompt 模板（ChatML 格式，306 字符）

```
<|im_start|>system
你是智能家居助手。根据用户指令，用一句话回复你执行了什么操作。

可用设备和操作：
- 客厅灯/卧室灯/厨房灯：打开、关闭、调亮度(0-100)
- 客厅空调/卧室空调：打开、关闭、调温度(16-30度)、调模式(制冷/制热/送风)
- 客厅窗帘/卧室窗帘：打开、关闭
- 电视：打开、关闭、调频道、调音量
- 净化器：打开、关闭、调模式
- 热水器：打开、关闭、调温度

回复格式：直接说执行了什么，如"客厅灯已打开，亮度80%"

用户说：<|im_end|>
<|im_start|>user
{command_text}<|im_end|>
<|im_start|>assistant
```

**Prompt 优化历程**：
- 初版：1400+ 字符，包含详细规则、示例、DEVICE|ACTION|PARAMS 管道格式
- 精简到 306 字符（4.5x 缩小），LLM 推理时间从 ~12s 降到 ~9s
- LLM 输出从结构化管道格式改为自然语言（"客厅灯已打开，亮度80%"）

### 6. USB 麦克风支持

#### 6.1 设备发现

设备支持两个 USB 音频输入：
1. **C-Media USB PnP Sound Device**（Card 0）：16000Hz, S16_LE, mono —— 桌面 USB 麦克风
2. **XIBERIA K03S**（Card 1）：48000Hz, S24_3LE, mono —— 游戏耳机

#### 6.2 C-Media 麦克风配置

```python
# 录制约 3 秒，16kHz, S16_LE, mono
arecord -D plughw:0,0 -d 3 -f S16_LE -r 16000 -c 1 output.wav
```

增益归一化：USB 麦克风始终满增益（peak=32768，削波），软件乘以 0.3 衰减。

#### 6.3 K03S 耳机麦克风

K03S 原生格式为 S24_3LE 48kHz。S16_LE 产生静音。必须在录音后软件降采样：

```python
# 24-bit (3 bytes/sample) → int16 → 3x decimate (48k→16k)
n = len(data) // 3
s16 = [0] * n
for i in range(n):
    b = data[i*3:(i+1)*3]
    val = b[0] | (b[1] << 8) | (b[2] << 16)
    if val & 0x800000: val -= 0x1000000
    s16[i] = val >> 8
decimated = s16[::3]
```

### 7. 3.5mm 内置音频输出

#### 7.1 硬件信息

SC171 板载 Qualcomm lahaina-yupikiot 音频 DSP，3.5mm 耳机孔连接到该 DSP。

#### 7.2 HPH（Headphone）音频路径配置

3.5mm 耳机输出需要配置 Qualcomm 音频 DSP 的 HPH 路径。使用 tinymix 工具（tinyalsa）：

```bash
# 音频路由：AIF1 → RX0/RX1 → CLSH_DSM_OUT → HPH
tinymix set 89 AIF1_PB      # AIF1 音频接口 → 播放路由
tinymix set 90 AIF1_PB
tinymix set 111 RX0          # RX0 接收通道
tinymix set 112 RX1          # RX1 接收通道
tinymix set 106 CLSH_DSM_OUT # DEM 调制器模式
tinymix set 107 CLSH_DSM_OUT
tinymix set 136 1            # HPH 开关
tinymix set 137 1
tinymix set 243 1
tinymix set 244 1
tinymix set 268 1
tinymix set 269 1
tinymix set 133 124          # 数字音量（最大 124）
tinymix set 134 124
tinymix set 519 1            # 全局使能（关键！）
```

**关键发现**：
- **tinymix 不带 `-D` 标志时 SET 命令正常工作**，默认操作 Card 0
- **tinymix 带 `-D N` 标志时 SET 命令全部失败**（"Error: 3 values given, but control only takes 1"）
- 控制 519（全局使能）必须设置，否则 `plughw` 拒绝硬件参数

#### 7.3 播放配置

```bash
# 显式指定 S16_LE/48000Hz/mono，使用大缓冲区避免 underrun
aplay -D plughw:1,5 -f S16_LE -r 48000 -c 1 \
      --buffer-size=65536 --period-size=4096 audio.wav
```

- **Card 1, device 0**（MultiMedia1）：需要 HPH 配置才能接受硬件参数
- **Card 1, device 5**（AFE-PROXY RX）：直接可用，无需 HPH 配置

### 8. 智能家居助手 Pipeline 架构

#### 8.1 项目结构

```
/home/fibo/project/HCI/
├── main.py                 # 入口点
├── config.py               # 配置（路径、模型、麦克风、唤醒词）
├── pipeline.py             # 状态机编排器
├── asr_engine.py           # Vosk 流式 ASR 封装
├── wake_word.py            # 流式唤醒词检测器
├── llm_engine.py           # llama.cpp 查询引擎
├── action_executor.py      # LLM 输出解析 + 动作执行
├── smart_home_state.py     # 设备状态存储（单例）
├── utils.py                # 音频工具、TTS、日志
├── prompt_templates.py     # LLM 提示模板
├── deploy.sh               # 部署脚本
├── sounds/
│   ├── ready.wav           # "智能家居助手已就绪"
│   └── wozai.wav           # "收到"
├── logs/
│   ├── pipeline.log
│   └── errors.log
└── output/                 # 调试音频文件
```

#### 8.2 处理流程

```
[启动] → 加载 Vosk 模型 → 加载 LLM → 播放 "智能家居助手已就绪" → 进入主循环
                                                                    │
   ┌────────────────────────────────────────────────────────────────┘
   │
   ▼
[Phase 1: 流式监听]
   arecord 持续录音 → Vosk 每 100ms 输出部分结果 → deque 滑动窗口
   → 检测 "小电"（模糊匹配：小电/小点/小店/笑电）
   │
   ├─ 未检测到 → 旧文本自动出队（deque FIFO），继续监听
   │
   └─ 检测到 → 提取 "小电" 后的文本作为指令
              → 回溯 3s 唤醒前音频 + 捕获最长 10s 唤醒后音频
              │
              ▼
[Phase 2: 播放 "收到"]
   play_sound(wozai.wav, blocking=True)  # 同步等待 0.8s
              │
              ▼
[Phase 3: 指令识别]
   Vosk 重新转录完整音频（含唤醒前 3s + 唤醒后最多 10s）
   → MaxAlternatives(3) 取置信度最高的候选
              │
              ▼
[Phase 4: LLM 解析]
   build_prompt(command_text) → ChatML 306 字符 prompt
   → llama-completion 子进程（~9s）
   → "客厅灯已打开，亮度80%"
              │
              ▼
[Phase 5: 执行 + TTS]
   parse_llm_output(llm_text) → 关键词提取 device/action/params
   → execute() → 更新 SmartHomeState
   → speak(llm_text) → espeak-ng TTS → aplay 播放
              │
              └──→ 回到 Phase 1
```

#### 8.3 唤醒词检测策略

正则表达式：`[小笑][\s]?[电点店垫典颠]`

- 允许可选空格（Vosk 输出带空格分词）
- 同音词模糊匹配（考虑到语音识别误差）
- 允许嵌入在句子中任意位置

测试用例：
| 输入 | 匹配 |
|------|------|
| `小电` | ✓ |
| `小点` | ✓ |
| `打开小电` | ✓ |
| `小电打开灯` | ✓（指令=`打开灯`） |

#### 8.4 缓冲区管理

**文本队列**（`deque(maxlen=40)`）：
- 3s 唤醒前音频对应的文本
- FIFO 自动清理旧条目
- Recognizer 18s 重置时清空

**音频环形缓冲区**（`deque`，最大 3s）：
- 保存唤醒前音频数据
- 唤醒时一起捕获，确保"小电"被包含在录音中

**唤醒后缓冲区**（`list`，最大 10s）：
- 捕获完整用户指令
- 上限 10s 防止内存膨胀

#### 8.5 Recognizer 生命周期管理

**核心问题**：Vosk decoder lattice 无限累积导致内存耗尽。

**解决方案**：每 18 秒创建新的 KaldiRecognizer，清空文本队列和音频环形缓冲区。

```python
RECOGNIZER_RESET_SEC = 18
if not wake_detected and (time.time() - rec_started) > RECOGNIZER_RESET_SEC:
    rec = _new_recognizer()
    text_queue.clear()
    audio_ring.clear()
```

#### 8.6 CPU 共享策略

最初设计使用 SIGSTOP/SIGCONT 冻结 llama.cpp 进程以共享 CPU：

```python
os.kill(proc.pid, signal.SIGSTOP)  # 冻结（0% CPU，模型保持在 RAM）
os.kill(proc.pid, signal.SIGCONT)  # 恢复
```

后来简化为按需调用（`subprocess.run`），因为：
1. Vosk ASR 和 LLM 推理是**串行**的（ASR 完成后才调用 LLM）
2. 模型从 OS 页面缓存加载只需 ~1-2s
3. 避免了复杂的管道管理和死锁风险

#### 8.7 智能家居设备模拟

10 个虚拟设备：

| 设备 | 类型 | 动作 |
|------|------|------|
| living_room_light | 灯 | 打开/关闭/调亮度 |
| bedroom_light | 灯 | 打开/关闭/调亮度 |
| kitchen_light | 灯 | 打开/关闭/调亮度 |
| living_room_ac | 空调 | 打开/关闭/调温度/调模式 |
| bedroom_ac | 空调 | 打开/关闭/调温度/调模式 |
| living_room_curtain | 窗帘 | 打开/关闭 |
| bedroom_curtain | 窗帘 | 打开/关闭 |
| tv | 电视 | 打开/关闭/调频道/调音量 |
| air_purifier | 净化器 | 打开/关闭/调模式 |
| water_heater | 热水器 | 打开/关闭/调温度 |

状态存储在 `SmartHomeState` 单例中，所有操作在内存中进行。

#### 8.8 音频设备 Card 布局变化

Qualcomm 设备在每次启动时 USB 设备的探测顺序不同，导致 Card 编号变化：

| 时期 | Card 0 | Card 1 | Card 2 |
|------|--------|--------|--------|
| 初始（10.0.0.102） | lahaina | C-Media | K03S |
| 重启 1（10.42.0.254） | lahaina | K03S | C-Media |
| 重启 2（10.42.0.72） | C-Media | lahaina | K03S |
| 重启 3（10.203.67.252） | lahaina | K03S | C-Media |
| 重启 4（10.203.67.137） | C-Media | lahaina | — |

配置文件使用 `plughw:X,0` 格式，需要根据实际 Card 编号调整。

### 9. 已验证的端到端流程

```
用户说："小电，打开客厅灯"
         │
         ▼
Vosk 流式识别 → "小 电 打开 客厅 灯"
         │
         ▼
唤醒检测：匹配 "小电"
         │
         ▼
回溯 3s + 捕获 10s 音频 → Vosk 重新转录 → "打开客厅灯"
         │
         ▼
播放 "收到"（同步，0.8s）
         │
         ▼
LLM（306 chars, ~9s）→ "客厅灯已打开，亮度80%"
         │
         ▼
action_executor → living_room_light: turn_on, brightness=80
         │
         ▼
espeak-ng TTS → "客厅灯已打开，亮度百分之八十"
         │
         ▼
aplay → 3.5mm 音响输出
```

---

## 第二部分：存在的困难

### 1. GPU/DSP 硬件加速不可用

#### 1.1 Adreno 643 GPU

**困难描述**：Adreno 643 GPU 的 Vulkan 1.1.128 驱动版本过旧，无法满足现代推理框架（llama.cpp ggml-vulkan、MNN Vulkan）的 Vulkan ≥1.2 要求。GPU busy 计数器始终为 0，所有 Vulkan 推理静默回退到 CPU。

**影响**：LLM 推理完全依赖 CPU（8 核 ARM），无 GPU 加速可能。

#### 1.2 QNN DSP 版本不兼容

**困难描述**：设备上的 fiboaisdk 包含 QNN 2.29 运行时库，但所有预编译的 `.fmodel` 文件（Qwen3 0.6B LLM、fiboasr_base ASR、fibotts TTS）均编译为 QNN 2.26 或 2.28。DSP skeleton 版本不匹配导致所有模型加载失败（error 1008: `QNN_COMMON_ERROR_INCOMPATIBLE_BINARIES`）。

**影响**：无法使用 DSP 加速任何模型推理。DSP 是唯一能释放 CPU 的硬件加速器。

#### 1.3 QNN SDK Converter C 层 Bug

**困难描述**：QNN SDK 2.29.0 的 ONNX→DLC 转换器在 C 层形状推导中存在 bug。Cast 操作被"转换时解释"后，其输出张量的形状信息无法传播到后续的 Unsqueeze/Reshape 操作，导致转换失败。尝试了 6 种 ONNX 图修复方法（删除 Cast、替换为 Identity、替换 Unsqueeze→Reshape 等）均无法绕过。

**影响**：无法将新模型（如 Qwen2.5 0.5B 的 ONNX 导出）编译为 QNN GPU/DSP 格式。

### 2. 音频设备管理复杂

#### 2.1 USB 设备探测顺序不稳定

**困难描述**：Qualcomm 设备在每次启动时，USB 音频设备的 ALSA Card 编号会变化。C-Media 麦克风和 K03S 耳机可能在 Card 0、1、2 之间任意排列。配置文件中的 `plughw:X,0` 需要根据实际情况手动调整。

#### 2.2 tinymix ENUM 控制 Bug

**困难描述**：该设备上的 tinymix（tinyalsa v1.1.0）存在严重 bug：
- `tinymix set` 不带 `-D` 参数时正常工作（默认 Card 0）
- `tinymix set` 带 `-D N` 参数时 ENUM/BOOL/INT 全部失败（"Error: 3 values given, but control only takes 1"）
- 当 lahaina 音频 DSP 不是 Card 0 时，无法配置 HPH 耳机路径

**影响**：3.5mm 内置耳机孔需要 HPH 路径配置。当 lahaina 不是 Card 0 时，无法通过 tinymix 配置，导致内置音响不可用。

**解决方案尝试**：
1. libasound ctypes ioctl 直写：使用 `snd_ctl_elem_write` API 直接操作 Card 1 的 ALSA 控制。找到了正确的缓冲区大小（1088 字节）和 ioctl 编码，成功设置所有 HPH 控制。但设备 0 仍然拒绝硬件参数。
2. 使用 Card 1, device 5（AFE-PROXY RX）：绕过设备 0 的硬件参数限制，直接可用（rc=0）。

#### 2.3 K03S 麦克风 S24_3LE 格式限制

**困难描述**：K03S 耳机支持 S24_3LE 48kHz 原生格式，但 S16_LE 格式产生静音。需要软件进行 24-bit→16-bit 转换和 48kHz→16kHz 降采样，增加了处理复杂度。

#### 2.4 内置 3.5mm 耳机孔播放后 aplay 不退出

**困难描述**：aplay 通过 `plughw:1,5` 播放音频后，进程不退出（rc=124 超时）。音频数据已经成功发送到硬件（underrun 消息确认），但 aplay 等待硬件反馈。解决方案：使用 `timeout 4` 包装 aplay 调用。

### 3. Vosk 模型限制

#### 3.1 内存消耗

**困难描述**：Vosk 中文模型需要 ~700MB RAM（去掉 RNNLM 后）。加上 Python 运行时和其他组件，总内存使用约 800MB。在 8GB 设备上可行，但启动时模型加载时间 ~3s。

#### 3.2 Decoder Lattice 膨胀

**困难描述**：Vosk 的 KaldiRecognizer 在长时间连续识别时，decoder lattice 会无限增长，最终导致内存耗尽或识别停止。

**解决方案**：每 18 秒重置 recognizer，清空文本和音频缓冲区。代价是丢失跨重置边界的上下文。

#### 3.3 中文同音词歧义

**困难描述**：Vosk 对同音词的识别存在误差。例如唤醒词"小电"可能被识别为"小店""小点"等。解决方案是使用正则模糊匹配。

### 4. LLM 推理性能

#### 4.1 延迟

**困难描述**：Qwen2.5 0.5B GGUF 模型在处理 306 字符的系统提示时，从冷启动到输出需要 ~9s（含提示处理 ~35 tok/s + 生成 ~13 tok/s）。这个延迟在智能家居场景中可接受（用户说完指令到系统执行约 10-15s 总延迟），但仍有提升空间。

#### 4.2 精度限制

**困难描述**：0.5B 参数的模型在理解复杂指令时精度有限。例如"把空调调到 26 度"有时会被误解为"打开客厅灯并设置温度 26"。解决方案：
1. 在 prompt 中提供 1-shot 示例
2. 使用关键词后备解析器处理 LLM 输出失败的情况
3. 降低温度参数（0.1）提高输出确定性

### 5. TTS 限制

#### 5.1 fiboaisdk TTS 不可用

**困难描述**：设备上的 fiboaisdk TTS 模型（`fibotts_1.0.0_qcom_6490-8550_qnn_2.26_dsp`）需要 QNN 2.26 DSP，与设备 QNN 2.29 不兼容。模型内部硬编码了 Android 路径（`/data/local/tmp/jinxu-android-qcom-aarch64/`）和开发机器的路径（`/home1/liujinxu/`），无法在 Ubuntu 设备上运行。

#### 5.2 espeak-ng 音质

**困难描述**：espeak-ng 是合成语音（formant-based），音质不如神经 TTS 模型自然。但优点是轻量（~1MB）、快速（毫秒级生成）、支持中文。

### 6. 网络连接不稳定

**困难描述**：SC171 设备通过 USB 网络共享连接，IP 地址经常变化（观察到的 IP：10.0.0.102, 10.42.0.254, 10.42.0.72, 10.203.67.252, 10.203.67.137）。每次重新连接需要手动发现新 IP。WireGuard 隧道（temp6.conf）提供备用连接路径但也不总是可用。

### 7. 麦克风信号质量

#### 7.1 USB 麦克风削波

**困难描述**：C-Media USB 麦克风始终输出满量程信号（peak=32768），即使用 amixer 调到最低（1/16 = 6%），peak 仍为 32768。这说明 USB 麦克风的增益控制在硬件层面无效，需要软件增益归一化（乘以 0.3）。

#### 7.2 K03S 信号极弱

**困难描述**：K03S 耳机麦克风的 24-bit 信号峰值仅约 1% 满量程（peak≈111,437/8,388,607），需要 20-30x 软件放大。自适应增益需要平衡信号放大和噪声抑制。

### 8. 遗留问题

1. **libasound HPH 配置未集成到最终版本**：在回退操作中移除了 `_setup_hph()` 方法。当前使用 Card 1 device 5 绕过 HPH 配置需求，但设备 0（MultiMedia1）仍然无法使用。

2. **Card 编号不稳定**：配置文件使用硬编码的 Card 编号，每次设备重启可能需要手动调整。

3. **Whisper Small OOM**：359MB 的 Whisper Small 模型在推理时因额外内存消耗（约 2GB）导致 OOM，无法使用更高精度的 ASR 模型。

4. **QNN 2.26/2.28 模型无法使用**：设备上有多个预编译的 `.fmodel` 文件（LLM、ASR、TTS），但均因 QNN 版本不匹配无法加载。

5. **PulseAudio 自动重启**：PulseAudio 被 systemd/dbus 自动重启，可能导致音频设备被占用。使用 `systemctl mask` 禁用后仍然会重启。

---

## 总结

本项目成功在 Qualcomm QCS6490 嵌入式设备上构建了一套完整的语音控制智能家居系统，包含：

- **流式中文语音识别**（Vosk Kaldi CN，2.7x 实时速度）
- **大语言模型指令解析**（llama.cpp Qwen2.5 0.5B，~9s 推理）
- **文本转语音反馈**（espeak-ng，中文 TTS）
- **流式唤醒词检测**（"小电"，模糊匹配）
- **音频捕获与播放**（USB 麦克风 + 3.5mm 内置耳机孔）
- **10 个虚拟智能家居设备模拟**

核心挑战在于 Adreno 643 GPU 无法用于推理加速，QNN DSP 模型版本不匹配，以及 Qualcomm 音频 DSP 的配置复杂性。尽管存在这些问题，系统在纯 CPU 模式下仍然能够正常工作，实现了完整的语音交互闭环。
