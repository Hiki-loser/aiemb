"""Audio normalization and logging utilities."""

import os
import sys
import wave
import struct
import logging
from datetime import datetime


def normalize_wav(wav_path, gain=0.3):
    """Apply software gain to a 16-bit mono WAV file in-place.
    USB mic clips at 32768, so we reduce amplitude by gain factor."""
    with wave.open(wav_path, "rb") as w:
        params = w.getparams()
        nframes = w.getnframes()
        data = w.readframes(nframes)

    samples = list(struct.unpack(f"<{len(data)//2}h", data))
    samples = [max(-32768, min(32767, int(s * gain))) for s in samples]
    normalized = struct.pack(f"<{len(samples)}h", *samples)

    with wave.open(wav_path, "wb") as w:
        w.setparams(params)
        w.writeframes(normalized)


def rms_energy(wav_path):
    """Compute RMS energy of a 16-bit mono WAV. Returns float."""
    with wave.open(wav_path, "rb") as w:
        nframes = w.getnframes()
        if nframes == 0:
            return 0.0
        data = w.readframes(min(nframes, 16000 * 10))
    samples = struct.unpack(f"<{len(data)//2}h", data)
    total = sum(abs(s) for s in samples)
    return total / len(samples)


def setup_logging(log_dir, debug=False):
    """Configure loggers: file + console in debug mode."""
    os.makedirs(log_dir, exist_ok=True)

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    pl = logging.getLogger("pipeline")
    pl.setLevel(logging.DEBUG if debug else logging.INFO)
    ph = logging.FileHandler(os.path.join(log_dir, "pipeline.log"))
    ph.setFormatter(fmt)
    pl.addHandler(ph)

    # Console handler for real-time output in debug mode
    if debug:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        ch.setLevel(logging.DEBUG)
        pl.addHandler(ch)

    el = logging.getLogger("errors")
    el.setLevel(logging.WARNING)
    eh = logging.FileHandler(os.path.join(log_dir, "errors.log"))
    eh.setFormatter(fmt)
    el.addHandler(eh)

    return pl


def resample_wav(src_path, dst_path, target_rate):
    """Resample any WAV (16-bit or 24-bit) to target_rate Hz 16-bit mono."""
    # Read raw samples as int32
    with wave.open(src_path, "rb") as w:
        src_rate = w.getframerate()
        nch = w.getnchannels()
        sampwidth = w.getsampwidth()
        nframes = w.getnframes()
        data = w.readframes(nframes)

    if sampwidth == 2:
        samples = [struct.unpack("<h", data[i*2:(i+1)*2])[0] for i in range(nframes)]
    elif sampwidth == 3:
        samples = []
        for i in range(nframes):
            b = data[i*3:(i+1)*3]
            val = b[0] | (b[1] << 8) | (b[2] << 16)
            if val & 0x800000:
                val -= 0x1000000
            samples.append(val >> 8)  # 24→16 bit
    else:
        raise ValueError(f"Unsupported sampwidth: {sampwidth}")

    if src_rate == target_rate:
        resampled = samples
    else:
        ratio = src_rate / target_rate
        new_len = int(len(samples) / ratio)
        resampled = [samples[int(i * ratio)] for i in range(new_len)]

    with wave.open(dst_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(target_rate)
        w.writeframes(struct.pack(f"<{len(resampled)}h", *resampled))


def peak_amplitude(wav_path):
    """Return max absolute sample value in a 16-bit mono WAV."""
    with wave.open(wav_path, "rb") as w:
        nframes = w.getnframes()
        if nframes == 0:
            return 0
        data = w.readframes(min(nframes, 16000 * 10))
    samples = struct.unpack(f"<{len(data)//2}h", data)
    return max(abs(s) for s in samples)


# Traditional → Simplified Chinese mapping for ASR output
# Whisper model natively outputs Traditional; SDK OpenCC may not work
T2S_MAP = str.maketrans({
    "國":"国","際":"际","會":"会","動":"动","為":"为","對":"对","機":"机",
    "體":"体","學":"学","發":"发","開":"开","關":"关","門":"门","時":"时",
    "來":"来","這":"这","個":"个","們":"们","說":"说","過":"过","沒":"没",
    "還":"还","電":"电","話":"话","見":"见","車":"车","長":"长","頭":"头",
    "書":"书","買":"买","賣":"卖","氣":"气","愛":"爱","實":"实","後":"后",
    "裡":"里","麼":"么","樣":"样","點":"点","聽":"听","從":"从","應":"应",
    "戰":"战","報":"报","變":"变","視":"视","臺":"台","眾":"众","飛":"飞",
    "馬":"马","樂":"乐","當":"当","經":"经","種":"种","萬":"万","線":"线",
    "遠":"远","語":"语","認":"认","讓":"让","進":"进","連":"连","間":"间",
    "問":"问","網":"网","節":"节","寫":"写","業":"业","產":"产","醫":"医",
    "歷":"历","壓":"压","嚴":"严","觀":"观","團":"团","爾":"尔","親":"亲",
    "現":"现","與":"与","總":"总","統":"统","華":"华","風":"风","龍":"龙",
    "衛":"卫","標":"标","選":"选","歐":"欧","權":"权","驗":"验","質":"质",
    "殺":"杀","導":"导","層":"层","帶":"带","歲":"岁","廠":"厂","廣":"广",
    "張":"张","藝":"艺","腦":"脑","舊":"旧","藥":"药","裝":"装","護":"护",
    "議":"议","讀":"读","變":"变","軍":"军","輪":"轮","辦":"办","農":"农",
    "運":"运","達":"达","鄉":"乡","鐵":"铁","銀":"银","際":"际","難":"难",
    "雲":"云","飯":"饭","魚":"鱼","鳥":"鸟","黃":"黄","黑":"黑","黨":"党",
    "園":"园","圓":"圆","圖":"图","塊":"块","聲":"声","處":"处","號":"号",
    "術":"术","製":"制","複":"复","規":"规","覺":"觉","計":"计","記":"记",
    "許":"许","試":"试","註":"注","證":"证","評":"评","話":"话","該":"该",
    "詳":"详","試":"试","誤":"误","說":"说","調":"调","論":"论","請":"请",
    "誰":"谁","謝":"谢","識":"识","譯":"译","議":"议","護":"护","讀":"读",
    "變":"变","讓":"让","貝":"贝","負":"负","貴":"贵","買":"买","費":"费",
    "資":"资","質":"质","賽":"赛","轉":"转","輕":"轻","較":"较","輛":"辆",
    "輸":"输","辦":"办","農":"农","連":"连","週":"周","過":"过","遊":"游",
    "運":"运","達":"达","遠":"远","選":"选","還":"还","邊":"边","郵":"邮",
    "鄉":"乡","鄰":"邻","鐵":"铁","長":"长","門":"门","間":"间","開":"开",
    "關":"关","隊":"队","際":"际","電":"电","雲":"云","靈":"灵","靜":"静",
    "頁":"页","頂":"顶","項":"项","順":"顺","須":"须","頭":"头","題":"题",
    "風":"风","飛":"飞","馬":"马","魚":"鱼","鳥":"鸟","龍":"龙","龜":"龟",
})


def t2s_convert(text):
    """Convert Traditional Chinese to Simplified. Returns unchanged if no mapping."""
    if not text:
        return text
    return text.translate(T2S_MAP)


def ts():
    """Timestamp string for filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def speak(text, lang="zh", blocking=False):
    """Generate TTS WAV via espeak-ng (48kHz) and play through speaker."""
    import subprocess, threading, os, wave, struct
    import config

    raw_path = f"/tmp/tts_raw_{datetime.now().strftime('%H%M%S%f')}.wav"
    out_path = f"/tmp/tts_{datetime.now().strftime('%H%M%S%f')}.wav"

    subprocess.run(
        ["espeak-ng", "-v", lang, "-w", raw_path, text],
        capture_output=True, timeout=5,
    )
    if not os.path.exists(raw_path) or os.path.getsize(raw_path) < 100:
        return

    # Resample to 48kHz for internal audio
    try:
        with wave.open(raw_path, "rb") as w:
            p = w.getparams()
            d = w.readframes(w.getnframes())
        samples = struct.unpack(f"<{len(d)//2}h", d)
        ratio = p.framerate / 48000
        new = [samples[int(i * ratio)] for i in range(int(len(samples) / ratio))]
        with wave.open(out_path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(48000)
            w.writeframes(struct.pack(f"<{len(new)}h", *new))
    except Exception:
        out_path = raw_path  # Fallback

    def _play():
        subprocess.run(
            ["timeout", "6", "aplay", "-D", config.SPEAKER_DEVICE,
             "-f", "S16_LE", "-r", "48000", "-c", "1",
             "--buffer-size=65536", "--period-size=4096", out_path],
            capture_output=True,
        )
        try:
            os.remove(raw_path)
        except:
            pass
        try:
            os.remove(out_path)
        except:
            pass

    if blocking:
        _play()
    else:
        threading.Thread(target=_play, daemon=True).start()
