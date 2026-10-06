#!/usr/bin/env python3
"""Quick test: does vkCreateDevice crash on Adreno 643?"""
import ctypes, os
os.environ["RUNTIME_LIB_PATHS"] = "/usr/local/lib/python3.8/dist-packages/fiboaisdk"

vk = ctypes.CDLL("libvulkan.so.1")

# Setup functions
vk.vkCreateInstance.restype = ctypes.c_int
vk.vkEnumeratePhysicalDevices.restype = ctypes.c_int
vk.vkGetPhysicalDeviceQueueFamilyProperties.restype = None
vk.vkCreateDevice.restype = ctypes.c_int
vk.vkDestroyDevice.restype = None
vk.vkDestroyInstance.restype = None

# Create instance
api_ver = 0x00401000  # Vulkan 1.1
app = (0, None, b"test", 0, b"eng", 0, api_ver)
info = (1, None, 0, ctypes.pointer(ctypes.py_object(app)), 0, None, 0, None)

instance = ctypes.c_void_p()
r = vk.vkCreateInstance(ctypes.pointer(ctypes.py_object(info)), None, ctypes.pointer(instance))
assert r == 0, f"vkCreateInstance failed: {r}"
print(f"Instance OK")

# Get physical device
n = ctypes.c_uint32(0)
vk.vkEnumeratePhysicalDevices(instance, ctypes.pointer(n), None)
gpus = (ctypes.c_uint64 * n.value)()
vk.vkEnumeratePhysicalDevices(instance, ctypes.pointer(n), gpus)
gpu = ctypes.c_void_p(gpus[0])
print(f"GPU handle: {gpu.value:#x}")

# Get queue families
nqf = ctypes.c_uint32(0)
vk.vkGetPhysicalDeviceQueueFamilyProperties(gpu, ctypes.pointer(nqf), None)
print(f"Queue families: {nqf.value}")

qfs = (ctypes.c_uint32 * (nqf.value * 4))()  # simplified: 4 uint32_t per QF
vk.vkGetPhysicalDeviceQueueFamilyProperties(gpu, ctypes.pointer(nqf), qfs)

# Find compute queue
qf_idx = 0
for i in range(nqf.value):
    flags = qfs[i*4]
    if flags & 0x02:  # COMPUTE bit
        qf_idx = i
        break

print(f"Using QF {qf_idx}")

# Create device - THE CRITICAL TEST
# VkDeviceQueueCreateInfo: sType=2, flags=0, qfIdx, count=1, pPriorities
prio = ctypes.c_float(1.0)
qci = (ctypes.c_uint32 * 6)(2, 0, 0, qf_idx, 1, ctypes.addressof(prio))

# VkDeviceCreateInfo: sType=3, flags=0, qciCount=1, pQci, ...
dci = (ctypes.c_uint64 * 8)(
    3, 0, 0,  # sType, pNext, flags
    1, ctypes.addressof(qci),  # queueCreateInfoCount, pQci
    0, 0,  # layer count, ppLayers
    0, 0,  # extension count, ppExtensions
)

print("Calling vkCreateDevice...")
device = ctypes.c_void_p()
r = vk.vkCreateDevice(gpu, ctypes.pointer(ctypes.py_object(dci)), None, ctypes.pointer(device))

if r == 0:
    print(f"SUCCESS! Device: {device.value:#x}")
    vk.vkDestroyDevice(device, None)
else:
    print(f"FAILED: {r}")

vk.vkDestroyInstance(instance, None)
