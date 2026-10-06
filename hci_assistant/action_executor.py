"""Parse LLM natural-language response and execute smart home actions.

LLM 输出自然语言 → 关键词提取 device/action/params → 更新设备状态.
"""

import re
import logging

import smart_home_state

_log = logging.getLogger("pipeline")


def parse_llm_output(raw):
    """从 LLM 的自然语言回复中提取设备、动作和参数.

    LLM 回复示例: "客厅灯已打开，亮度80%"
    提取: device=living_room_light, action=turn_on, params={brightness:80}

    解析策略:
      1. 设备: 关键词匹配 (客厅灯→living_room_light 等)
      2. 动作: 动词匹配 (打开/关闭/调高 等)
      3. 参数: 正则提取 (NN度→temperature, 亮度NN→brightness)
    """
    raw = raw.strip().replace("[end of text]", "").replace("<|im_end|>", "")

    result = {"device": None, "action": "unknown", "params": {}}

    # 设备关键词 → device_id 映射表
    device_map = [
        ("客厅灯", "living_room_light"), ("客厅灯光", "living_room_light"),
        ("卧室灯", "bedroom_light"), ("卧室灯光", "bedroom_light"),
        ("厨房灯", "kitchen_light"), ("厨房灯光", "kitchen_light"),
        ("客厅空调", "living_room_ac"), ("空调", "living_room_ac"),
        ("卧室空调", "bedroom_ac"),
        ("客厅窗帘", "living_room_curtain"), ("窗帘", "living_room_curtain"),
        ("卧室窗帘", "bedroom_curtain"),
        ("电视", "tv"), ("电视机", "tv"),
        ("净化器", "air_purifier"), ("空气净化器", "air_purifier"),
        ("热水器", "water_heater"),
    ]

    for keyword, dev_id in device_map:
        if keyword in raw:
            result["device"] = dev_id
            break

    # 动作检测 (动词 → action 映射)
    if any(w in raw for w in ["打开", "开启", "开灯", "开了"]):
        result["action"] = "turn_on"
    elif any(w in raw for w in ["关闭", "关掉", "关了"]):
        result["action"] = "turn_off"
    elif "打开" in raw and "窗帘" in raw:
        result["action"] = "open"
    elif any(w in raw for w in ["关上窗帘", "拉上窗帘", "关闭窗帘"]):
        result["action"] = "close"

    # 参数提取 (正则匹配)
    # 温度: "26度" → temperature=26
    temp_match = re.search(r'(\d+)\s*度', raw)
    if temp_match:
        result["params"]["temperature"] = temp_match.group(1)

    # 亮度: "亮度80" → brightness=80
    bright_match = re.search(r'亮度\s*(\d+)', raw)
    if bright_match:
        result["params"]["brightness"] = bright_match.group(1)
    elif re.search(r'(百分[之]?)(\d+)', raw):
        result["params"]["brightness"] = re.search(r'(百分[之]?)(\d+)', raw).group(2)

    # 模式: "模式:制冷" → mode=cool
    mode_match = re.search(r'模式[：:]\s*(\S+)', raw)
    if mode_match:
        result["params"]["mode"] = mode_match.group(1)

    return result


def execute(parsed):
    """执行解析后的动作, 返回中文反馈字符串.
    更新 SmartHomeState 单例 (设备状态持久化在内存中).
    """
    device = parsed.get("device")
    action = parsed.get("action", "")
    params = parsed.get("params", {})

    if not device or action == "unknown":
        return "抱歉，我没有理解你的指令"

    dev = smart_home_state.DEVICE_REGISTRY.get(device)
    if not dev:
        return f"未知设备: {device}"

    dev_type = dev["type"]
    state = smart_home_state.SmartHomeState()  # 单例, 多次调用返回同一实例
    label = dev.get("label", device)

    try:
        if action == "turn_on":
            state.set(device, "state", "on")
            if "brightness" in params and dev_type == "light":
                state.set(device, "brightness", int(params["brightness"]))
            if "temperature" in params and dev_type in ("ac", "heater"):
                state.set(device, "temperature", int(params["temperature"]))
            return f"{label}：已开启"

        elif action == "turn_off":
            state.set(device, "state", "off")
            return f"{label}：已关闭"

        elif action == "set_temperature":
            val = int(params.get("temperature", 24))
            val = max(16, min(30, val))    # 安全范围 16-30°C
            state.set(device, "temperature", val)
            if dev_type == "ac":
                state.set(device, "state", "on")  # 调温隐含开机
            return f"{label}：温度已调至{val}度"

        elif action == "set_brightness":
            val = int(params.get("brightness", 80))
            val = max(1, min(100, val))    # 安全范围 1-100%
            state.set(device, "state", "on")
            state.set(device, "brightness", val)
            return f"{label}：亮度已调至{val}%"

        elif action == "open":
            state.set(device, "state", "open")
            return f"{label}：已打开"

        elif action == "close":
            state.set(device, "state", "closed")
            return f"{label}：已关闭"

        elif action == "set_mode":
            mode = params.get("mode", "auto")
            state.set(device, "mode", mode)
            state.set(device, "state", "on")
            return f"{label}：模式已设为{mode}"

        else:
            return f"{label}：操作成功"

    except (ValueError, TypeError) as e:
        return f"{label}：参数错误 - {e}"
