"""SC171 HPH Speaker driver using tinyalsa C library.
tinymix CLI tool is broken on this board, so we use the library directly."""

import ctypes
import subprocess
import os


_lib = None
_mixer = None


def _get_lib():
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL("libtinyalsa.so.1")
        _lib.mixer_open.argtypes = [ctypes.c_uint]
        _lib.mixer_open.restype = ctypes.c_void_p
        _lib.mixer_close.argtypes = [ctypes.c_void_p]
        _lib.mixer_get_ctl_by_name.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        _lib.mixer_get_ctl_by_name.restype = ctypes.c_void_p
        _lib.mixer_ctl_get_num_values.argtypes = [ctypes.c_void_p]
        _lib.mixer_ctl_get_num_values.restype = ctypes.c_uint
        _lib.mixer_ctl_set_value.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_int]
        _lib.mixer_ctl_set_value.restype = ctypes.c_int
    return _lib


def _set_ctl(name, val):
    """Set a mixer control by name to value. Returns True on success."""
    lib = _get_lib()
    ctl = lib.mixer_get_ctl_by_name(_mixer, name.encode())
    if not ctl:
        return False
    nv = lib.mixer_ctl_get_num_values(ctl)
    for i in range(nv):
        lib.mixer_ctl_set_value(ctl, i, val)
    return True


def init(volume=15):
    """Configure HPH audio path. Call once before playback."""
    global _mixer
    lib = _get_lib()
    _mixer = lib.mixer_open(2)  # Card 2 = internal lahaina-yupikiot

    vol = max(0, min(20, volume))
    gain = int(vol * 124 / 20)

    path = [
        ("RX_MACRO RX0 MUX", 1),
        ("RX_MACRO RX1 MUX", 1),
        ("RX INT0 DEM MUX", 1),
        ("RX INT1 DEM MUX", 1),
        ("RX INT0_1 MIX1 INP0", 5),
        ("RX INT1_1 MIX1 INP0", 6),
        ("RX_COMP1 Switch", 1),
        ("RX_COMP2 Switch", 1),
        ("HPHL_RDAC Switch", 1),
        ("HPHR_RDAC Switch", 1),
        ("HPHL Volume", vol),
        ("HPHR Volume", vol),
        ("RX_CDC_DMA_RX_0 Audio Mixer MultiMedia1", 1),
    ]
    for name, val in path:
        _set_ctl(name, val)
    for gctl in ["RX INT0_1 Gain", "RX INT1_1 Gain"]:
        _set_ctl(gctl, gain)


def play(wav_path, blocking=True):
    """Play a WAV file through the HPH 3.5mm jack."""
    if not os.path.exists(wav_path):
        return False
    cmd = ["aplay", "-D", "hw:2,0", wav_path]
    if blocking:
        subprocess.run(cmd, capture_output=True, timeout=30)
    else:
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return True


def beep(freq=440, duration=0.3, volume=15):
    """Play a simple sine beep."""
    import math, struct, wave, tempfile
    sr = 48000
    n = int(sr * duration)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp = f.name
    with wave.open(tmp, "w") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(sr)
        for i in range(n):
            t = i / sr
            env = max(0, 1 - t / duration)
            val = int(8000 * env * math.sin(2 * math.pi * freq * t))
            w.writeframes(struct.pack("<hh", val, val))
    init(volume)
    play(tmp)
    os.unlink(tmp)


def close():
    global _mixer
    if _mixer:
        _get_lib().mixer_close(_mixer)
        _mixer = None
