"""Inspect the server's real Vulkan devices without installing Isaac Sim."""
import ctypes as c
import json


class InstanceInfo(c.Structure):
    _fields_ = [('sType', c.c_int), ('pNext', c.c_void_p), ('flags', c.c_uint32),
                ('pApplicationInfo', c.c_void_p), ('layer_count', c.c_uint32),
                ('layers', c.c_void_p), ('extension_count', c.c_uint32),
                ('extensions', c.c_void_p)]


class Extension(c.Structure):
    _fields_ = [('name', c.c_char * 256), ('version', c.c_uint32)]


def main():
    vk = c.CDLL('libvulkan.so.1')
    vk.vkCreateInstance.argtypes = [c.POINTER(InstanceInfo), c.c_void_p, c.POINTER(c.c_void_p)]
    vk.vkEnumeratePhysicalDevices.argtypes = [c.c_void_p, c.POINTER(c.c_uint32), c.c_void_p]
    vk.vkGetPhysicalDeviceProperties.argtypes = [c.c_void_p, c.c_void_p]
    vk.vkEnumerateDeviceExtensionProperties.argtypes = [
        c.c_void_p, c.c_char_p, c.POINTER(c.c_uint32), c.c_void_p]
    vk.vkDestroyInstance.argtypes = [c.c_void_p, c.c_void_p]
    instance = c.c_void_p()
    status = vk.vkCreateInstance(c.byref(InstanceInfo(sType=1)), None, c.byref(instance))
    if status:
        raise RuntimeError(f'vkCreateInstance returned {status}')
    try:
        count = c.c_uint32()
        if vk.vkEnumeratePhysicalDevices(instance, c.byref(count), None):
            raise RuntimeError('Cannot enumerate Vulkan devices')
        devices = (c.c_void_p * count.value)()
        if vk.vkEnumeratePhysicalDevices(instance, c.byref(count), devices):
            raise RuntimeError('Cannot read Vulkan devices')
        result = []
        for device in devices:
            properties = (c.c_uint64 * 1024)()
            vk.vkGetPhysicalDeviceProperties(device, properties)
            vendor = c.cast(properties, c.POINTER(c.c_uint32))[2]
            name = c.string_at(c.addressof(properties) + 20).decode()
            n = c.c_uint32()
            if vk.vkEnumerateDeviceExtensionProperties(device, None, c.byref(n), None):
                raise RuntimeError(f'Cannot enumerate extensions on {name}')
            extensions = (Extension * n.value)()
            if vk.vkEnumerateDeviceExtensionProperties(device, None, c.byref(n), extensions):
                raise RuntimeError(f'Cannot read extensions on {name}')
            names = {extension.name.decode() for extension in extensions}
            result.append({'device': name, 'vendor_id': hex(vendor),
                           'ray_tracing_pipeline': 'VK_KHR_ray_tracing_pipeline' in names,
                           'acceleration_structure': 'VK_KHR_acceleration_structure' in names})
        print(json.dumps(result, indent=2))
    finally:
        vk.vkDestroyInstance(instance, None)


if __name__ == '__main__':
    main()
