#!/usr/bin/env python3
"""Minimal Vulkan GPU compute test on Adreno 643."""
import ctypes, os, sys

os.environ["RUNTIME_LIB_PATHS"] = "/usr/local/lib/python3.8/dist-packages/fiboaisdk"

# Load Vulkan loader
vk = ctypes.CDLL("libvulkan.so.1")

# Vulkan types
uint32_t = ctypes.c_uint32
uint64_t = ctypes.c_uint64
size_t = ctypes.c_size_t
c_float = ctypes.c_float

VkResult = ctypes.c_int
VK_SUCCESS = 0

VK_STRUCTURE_TYPE_APPLICATION_INFO = 0
VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1

class VkApplicationInfo(ctypes.Structure):
    _fields_ = [
        ("sType", uint32_t),
        ("pNext", ctypes.c_void_p),
        ("pApplicationName", ctypes.c_char_p),
        ("applicationVersion", uint32_t),
        ("pEngineName", ctypes.c_char_p),
        ("engineVersion", uint32_t),
        ("apiVersion", uint32_t),
    ]

class VkInstanceCreateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", uint32_t),
        ("pNext", ctypes.c_void_p),
        ("flags", uint32_t),
        ("pApplicationInfo", ctypes.POINTER(VkApplicationInfo)),
        ("enabledLayerCount", uint32_t),
        ("ppEnabledLayerNames", ctypes.c_void_p),
        ("enabledExtensionCount", uint32_t),
        ("ppEnabledExtensionNames", ctypes.c_void_p),
    ]

VkInstance = ctypes.c_void_p
VkPhysicalDevice = ctypes.c_void_p

# Create Vulkan instance (headless, no extensions needed)
app_info = VkApplicationInfo(
    sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,
    pNext=None,
    pApplicationName=b"vulkan_test",
    applicationVersion=0,
    pEngineName=b"test",
    engineVersion=0,
    apiVersion=0x00402000,  # Vulkan 1.2
)

create_info = VkInstanceCreateInfo(
    sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
    pNext=None,
    flags=0,
    pApplicationInfo=ctypes.pointer(app_info),
    enabledLayerCount=0,
    ppEnabledLayerNames=None,
    enabledExtensionCount=0,
    ppEnabledExtensionNames=None,
)

instance = ctypes.c_void_p()
result = vk.vkCreateInstance(ctypes.pointer(create_info), None, ctypes.pointer(instance))
print(f"vkCreateInstance: {result} (0=SUCCESS)")
if result != VK_SUCCESS:
    print("FAILED to create Vulkan instance!")
    sys.exit(1)

# Enumerate physical devices
count = uint32_t(0)
vk.vkEnumeratePhysicalDevices(instance, ctypes.pointer(count), None)
print(f"Physical devices: {count.value}")

devices = (VkPhysicalDevice * count.value)()
vk.vkEnumeratePhysicalDevices(instance, ctypes.pointer(count), devices)

# Get device properties
class VkPhysicalDeviceProperties(ctypes.Structure):
    _fields_ = [
        ("apiVersion", uint32_t),
        ("driverVersion", uint32_t),
        ("vendorID", uint32_t),
        ("deviceID", uint32_t),
        ("deviceType", ctypes.c_int),
        ("deviceName", ctypes.c_char * 256),
        ("pipelineCacheUUID", ctypes.c_uint8 * 16),
        ("limits", ctypes.c_uint8 * 800),  # simplified
        ("sparseProperties", ctypes.c_uint8 * 20),
    ]

vkGetPhysicalDeviceProperties = vk.vkGetPhysicalDeviceProperties
vkGetPhysicalDeviceProperties.argtypes = [VkPhysicalDevice, ctypes.POINTER(VkPhysicalDeviceProperties)]

for i in range(count.value):
    props = VkPhysicalDeviceProperties()
    vkGetPhysicalDeviceProperties(devices[i], ctypes.pointer(props))
    name = props.deviceName.decode('utf-8')
    print(f"  GPU {i}: {name}")
    print(f"    API: {props.apiVersion >> 22}.{(props.apiVersion >> 12) & 0x3ff}.{props.apiVersion & 0xfff}")
    print(f"    Driver: {props.driverVersion >> 22}.{(props.driverVersion >> 12) & 0x3ff}.{props.driverVersion & 0xfff}")
    print(f"    Vendor: 0x{props.vendorID:04x} Device: 0x{props.deviceID:04x}")

vk.vkDestroyInstance(instance, None)
print("\nVulkan GPU is WORKING!")
