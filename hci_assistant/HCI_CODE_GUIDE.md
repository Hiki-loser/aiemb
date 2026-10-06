# Smart Home HCI — 核心代码注释指南

## 项目文件结构

```
/home/fibo/project/HCI/           # 设备端部署路径
├── main.py                        # 入口 (44行)
├── config.py                      # 全部配置 (68行)
├── pipeline.py                    # 主控状态机 (150行)
├── wake_word.py                   # 流式唤醒词检测 (230行)
├── asr_engine.py                  # Vosk 模型封装 (90行)
├── llm_engine.py                  # llama.cpp 子进程管理 (90行)
├── prompt_templates.py            # LLM Prompt (31行)
├── action_executor.py             # 指令解析 + 动作执行 (115行)
├── smart_home_state.py            # 设备状态单例 (130行)
├── utils.py                       # 音频处理/TTS/日志 (250行)
├── sounds/
│   ├── ready.wav                  # 启动音 "智能家居助手已就绪"
│   ├── wozai.wav                  # 唤醒响应 "收到"
│   └── ding.wav                   # 提示音
├── logs/                          # 运行时日志
└── output/                        # 调试产物
```

---

## pipeline.py — 主控状态机

```
启动: _setup_hph() → Vosk加载 → LLM就绪 → 播放启动音 → 进入主循环
主循环:
  wake.listen()           # 流式录音 + 唤醒词检测
  play_sound("收到")       # 同步播放 (阻塞等待完成)
  build_prompt(指令)       # 构建 306 字符 ChatML prompt
  llm.query(prompt)       # 调用 llama.cpp 推理
  parse_llm_output()      # 从自然语言回复中提取设备/动作/参数
  execute(parsed)          # 执行动作, 更新 SmartHomeState
  speak(llm_output)        # TTS 播报 (非阻塞)
  → 回到 wake.listen()
```

### play_sound() — 音频播放

```python
def play_sound(path, blocking=False):
    cmd = ["timeout", "6", "aplay", "-D", config.SPEAKER_DEVICE,
           "-f", "S16_LE", "-r", "48000", "-c", "1",
           "--buffer-size=65536", "--period-size=4096", path]
```

- 显式指定 `-f S16_LE -r 48000 -c 1` 避免 hw params 错误
- `blocking=True` 用于启动音/唤醒响应 (同步)
- `blocking=False` 用于 TTS 回复 (异步后台线程)

### _setup_hph() — 3.5mm 音频路径配置

通过 tinymix 设置 16 个控制寄存器:
1. AIF1 路由到 RX0/RX1 (89, 90)
2. DEM 调制器设置为 CLSH_DSM_OUT (106, 107)
3. RX 输入通道选择 (111, 112)
4. 数字音量最大 (133, 134: 124/124)
5. HPH 开关使能 (136, 137, 243, 244, 268, 269)
6. HPH 模式 CLS_H_HIFI (242)
7. 全局使能 **519 = 1** (关键! 缺失则 hw params 被拒)

---

## wake_word.py — 流式唤醒词检测

### 整体架构

```
arecord → 48000Hz S24_3LE/S16_LE → 软件降采样到 16000Hz
  → Vosk AcceptWaveform() 每 ~100ms 喂一次
  → PartialResult() 获取部分文本
  → deque(maxlen=40) 滑动窗口存储文本
  → regex 扫描整个窗口检测唤醒词
```

### 缓冲区管理

| 缓冲区 | 类型 | 大小 | 策略 |
|--------|------|------|------|
| `text_queue` | `deque(maxlen=40)` | ~4s 文本 | FIFO 自动出队 |
| `audio_ring` | `deque` | 3s 音频 | 手动弹出入, 唤醒时全部加入 post_wake |
| `post_wake_buf` | `list` | 最长 12s | 唤醒后捕获 (含回溯 3s + 实际 10s) |

### 两个关键路径

**快速路径** (用户说 "小电开灯" 一口气):
1. text_queue 拼接文本含 "小电开灯"
2. `after_wake = "开灯"` (≥4 字符)
3. 直接返回 "开灯" → 跳过音频捕获

**慢速路径** (用户说 "小电" 暂停再说指令):
1. text_queue 只有 "小电"
2. 回溯 audio_ring (含 "小电") → post_wake_buf
3. 继续录 10s → 重新转录 → 返回指令

### Recognizer 重置

```python
if not wake_detected and (time.time() - rec_started) > 18:
    rec = _new_recognizer()      # 新建 recognizer
    text_queue.clear()           # 清空文本队列
    audio_ring.clear()           # 清空音频环形缓冲
```

Vosk decoder lattice 在连续识别时无限增长。18 秒无唤醒则重置, 防止内存耗尽。

### 降采样 (K03S 48kHz → 16kHz)

```python
if mic["format"] == "S24_3LE":
    # 24-bit (3 bytes) → int16: 符号扩展 + 右移 8 位
    val = b[0] | (b[1] << 8) | (b[2] << 16)
    if val & 0x800000: val -= 0x1000000
    s16[i] = val >> 8
# 3x decimate: 48000 → 16000
decimated = s16[::3]
```

---

## action_executor.py — NLP 指令解析

### parse_llm_output() 解析流程

LLM 输出自然语言 → 三层提取:

**1. 设备提取** (关键词匹配):
```python
device_map = [
    ("客厅灯", "living_room_light"),
    ("空调", "living_room_ac"),
    ("窗帘", "living_room_curtain"),
    ...
]
```

**2. 动作提取** (动词匹配):
```python
"打开"/"开启"/"开灯" → turn_on
"关闭"/"关掉"       → turn_off
```

**3. 参数提取** (正则):
```python
(\d+)度           → temperature=N
亮度(\d+)         → brightness=N
模式[：:](\S+)    → mode=xxx
```

### execute() 动作执行

```python
def execute(parsed):
    state = SmartHomeState()  # 单例, 多次调用返回同一实例
    if action == "turn_on":
        state.set(device, "state", "on")
        ...
        return f"{label}：已开启"
```

每个动作执行后返回中文反馈字符串 (如 "客厅灯：已开启"), 同时更新内存中的设备状态。

---

## smart_home_state.py — 设备状态单例

### 单例模式

```python
class SmartHomeState:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
```

确保 `action_executor.execute()` 和 `pipeline.run()` 访问同一个设备状态实例。

### 设备注册表

```python
DEVICE_REGISTRY = {
    "living_room_light": {"type": "light", "state": "off", "brightness": 0, ...},
    "living_room_ac":    {"type": "ac",    "state": "off", "temperature": 24, "mode": "cool"},
    ...
}
```

每个设备包含 type/state 以及设备特有参数 (light 有 brightness, ac 有 temperature + mode 等)。

---

## llm_engine.py — LLM 推理

### 调用方式

```python
cmd = [
    LD_LINUX, LLAMA_BIN,
    "-m", MODEL, "-f", "/dev/stdin",
    "--temp", "0.1", "-n", "60", "-t", "4",
    "--cpu-mask", "0xFF",    # 全部 8 核
    "--no-conversation", "--no-display-prompt", "--single-turn",
]
proc = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=DEVNULL, env=env)
proc.stdin.write(prompt.encode())
proc.stdin.close()
proc.wait(timeout=30)
```

- `--single-turn`: 生成后自动退出 (不进入交互模式)
- `-f /dev/stdin`: 从 stdin 读取 prompt (避免 shell 转义问题)
- `--cpu-mask 0xFF`: 使用全部 8 核 (~9s), 仅 A55 可用 0x0F (~30s)
- `glibc 2.31` 兼容: 通过 `/home/fibo/newlibc/ld-linux-aarch64.so.1` 动态链接器

---

## utils.py — 工具函数

### speak() — TTS 语音合成

```python
def speak(text, lang="zh", blocking=False):
    # 1. espeak-ng 生成 WAV (22050Hz)
    espeak-ng -v zh -w raw.wav text
    # 2. 重采样到 48000Hz S16_LE
    # 3. aplay 播放
```

### t2s_convert() — 简繁转换

170+ 字符的映射表 (`str.maketrans`), 用于 Whisper Tiny 输出的繁体→简体转换。

### resample_wav() — 音频重采样

支持 16-bit 和 24-bit (3 字节) 输入 → 16-bit 输出, 简单最近邻降采样。

---

## 配置参数速查表

| 参数 | 值 | 说明 |
|------|-----|------|
| `WAKE_PATTERN` | `[小笑][\s]?[电点店垫典颠]` | 唤醒词模糊匹配 |
| `WAKE_SOUND` | sounds/wozai.wav | "收到" (0.8s) |
| `STARTUP_SOUND` | sounds/ready.wav | "智能家居助手已就绪" (2.6s) |
| `SPEAKER_DEVICE` | plughw:0,0 | 3.5mm 内置接口 |
| `SPEAKER_RATE` | 48000 | 播放采样率 |
| `SAMPLE_RATE` | 16000 | ASR 采样率 |
| `MAX_POST_WAKE_SEC` | 10.0 | 唤醒后最长录音 |
| `PRE_WAKE_SEC` | 3 | 唤醒前回溯音频 |
| `RECOGNIZER_RESET_SEC` | 18 | Recognizer 重置周期 |
| `LLM_TEMP` | 0.1 | LLM 温度 (低=更确定) |
| `LLM_TOKENS` | 60 | LLM 最大生成 token |
