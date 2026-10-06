"""In-memory smart home device state store."""

DEVICE_REGISTRY = {
    "living_room_light": {
        "type": "light",
        "state": "off",
        "brightness": 0,
        "color": "warm_white",
        "label": "Living Room Light",
    },
    "bedroom_light": {
        "type": "light",
        "state": "off",
        "brightness": 0,
        "color": "warm_white",
        "label": "Bedroom Light",
    },
    "kitchen_light": {
        "type": "light",
        "state": "off",
        "brightness": 0,
        "color": "cool_white",
        "label": "Kitchen Light",
    },
    "living_room_ac": {
        "type": "ac",
        "state": "off",
        "temperature": 24,
        "mode": "cool",
        "label": "Living Room AC",
    },
    "bedroom_ac": {
        "type": "ac",
        "state": "off",
        "temperature": 24,
        "mode": "cool",
        "label": "Bedroom AC",
    },
    "living_room_curtain": {
        "type": "curtain",
        "state": "closed",
        "label": "Living Room Curtain",
    },
    "bedroom_curtain": {
        "type": "curtain",
        "state": "closed",
        "label": "Bedroom Curtain",
    },
    "tv": {
        "type": "tv",
        "state": "off",
        "channel": 1,
        "volume": 20,
        "label": "TV",
    },
    "air_purifier": {
        "type": "purifier",
        "state": "off",
        "mode": "auto",
        "label": "Air Purifier",
    },
    "water_heater": {
        "type": "heater",
        "state": "off",
        "temperature": 45,
        "label": "Water Heater",
    },
}

VALID_ACTIONS = {
    "light": ["turn_on", "turn_off", "set_brightness", "set_color"],
    "ac": ["turn_on", "turn_off", "set_temperature", "set_mode"],
    "curtain": ["open", "close"],
    "tv": ["turn_on", "turn_off", "set_channel", "set_volume"],
    "purifier": ["turn_on", "turn_off", "set_mode"],
    "heater": ["turn_on", "turn_off", "set_temperature"],
    "all": ["query"],  # query valid for all types
}


class SmartHomeState:
    """Thread-safe (GIL) device state store."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_devices"):
            return  # Already initialized (singleton)
        self._devices = {}
        for dev_id, attrs in DEVICE_REGISTRY.items():
            self._devices[dev_id] = dict(attrs)

    def get(self, device_id):
        return self._devices.get(device_id)

    def set(self, device_id, key, value):
        if device_id not in self._devices:
            return False, f"Unknown device: {device_id}"
        dev = self._devices[device_id]
        dev[key] = value
        return True, f"{device_id}.{key} = {value}"

    def update(self, device_id, updates):
        if device_id not in self._devices:
            return False, f"Unknown device: {device_id}"
        self._devices[device_id].update(updates)
        return True, ""

    def get_all_states(self):
        """Return formatted state summary for LLM prompt."""
        lines = []
        for dev_id, dev in self._devices.items():
            if dev["type"] == "light":
                lines.append(
                    f"- {dev_id}: {dev['state']} "
                    f"(brightness={dev['brightness']}, color={dev['color']})"
                )
            elif dev["type"] == "ac":
                lines.append(
                    f"- {dev_id}: {dev['state']} "
                    f"(temperature={dev['temperature']}C, mode={dev['mode']})"
                )
            elif dev["type"] == "tv":
                lines.append(
                    f"- {dev_id}: {dev['state']} "
                    f"(channel={dev['channel']}, volume={dev['volume']})"
                )
            elif dev["type"] == "purifier":
                lines.append(
                    f"- {dev_id}: {dev['state']} (mode={dev['mode']})"
                )
            elif dev["type"] == "heater":
                lines.append(
                    f"- {dev_id}: {dev['state']} "
                    f"(temperature={dev['temperature']}C)"
                )
            else:
                lines.append(f"- {dev_id}: {dev['state']}")
        return "\n".join(lines)

    def describe(self, device_id):
        """Human-readable description of a device's current state."""
        dev = self._devices.get(device_id)
        if not dev:
            return f"Unknown device: {device_id}"
        label = dev.get("label", device_id)
        if dev["type"] == "light":
            return (
                f"{label} is {dev['state']}"
                f"{', brightness=' + str(dev['brightness']) if dev['state'] == 'on' else ''}"
                f", color={dev['color']}"
            )
        elif dev["type"] == "ac":
            return (
                f"{label} is {dev['state']}"
                f", temperature={dev['temperature']}C, mode={dev['mode']}"
            )
        elif dev["type"] == "curtain":
            return f"{label} is {dev['state']}"
        elif dev["type"] == "tv":
            return (
                f"{label} is {dev['state']}"
                f"{', channel=' + str(dev['channel']) if dev['state'] == 'on' else ''}"
                f", volume={dev['volume']}"
            )
        elif dev["type"] == "purifier":
            return f"{label} is {dev['state']}, mode={dev['mode']}"
        elif dev["type"] == "heater":
            return (
                f"{label} is {dev['state']}"
                f"{', temperature=' + str(dev['temperature']) + 'C' if dev['state'] == 'on' else ''}"
            )
        return f"{label}: {dev}"
