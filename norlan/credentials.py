"""Ark credentials, kept out of files, argv and the agent's shell environment.

Only explicit configuration writes to the current user's macOS Keychain.  The
Security framework receives secret data in memory, never through a subprocess.
Reads and health checks do not display authentication prompts in a background
production task.  No credential (or missing-credential result) is cached.
"""

from contextlib import contextmanager
import ctypes
import os
import sys


KEYCHAIN_SERVICE = "nolan.ark-api-key"
KEYCHAIN_ACCOUNT = "ark"
DEFAULT_ARK_ENV = "ARK_API_KEY"
MAX_KEY_BYTES = 8192

_SUCCESS = 0
_DUPLICATE_ITEM = -25299
_ITEM_NOT_FOUND = -25300
_UTF8 = 0x08000100


class CredentialError(RuntimeError):
    """An actionable, secret-free credential error."""

    def __init__(self, message, *, status=None, reason="unavailable"):
        super().__init__(message)
        self.status = status
        self.reason = reason


def _validate_key(value):
    # Reject control characters *before* stripping, including pasted newlines.
    if not isinstance(value, str) or any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise CredentialError("API Key 格式无效：不能包含换行或控制字符。", reason="invalid")
    value = value.strip()
    try:
        encoded_length = len(value.encode("utf-8"))
    except UnicodeError:
        raise CredentialError("API Key 格式无效：请重新粘贴完整密钥。", reason="invalid") from None
    if not value or encoded_length > MAX_KEY_BYTES:
        raise CredentialError("API Key 不能为空或超过 8192 字节。", reason="invalid")
    return value


def _error(status):
    status = int(status)
    if status in (-25308, -25315):
        message = "macOS 钥匙串已锁定或需要授权；请在“钥匙串访问”中解锁登录钥匙串，再运行 nolan ark-login。"
        reason = "locked"
    elif status in (-25293, -128):
        message = "macOS 钥匙串访问被拒绝或取消；请在“钥匙串访问”中检查授权，再运行 nolan ark-login。"
        reason = "locked"
    elif status in (-25291, -25294, -25295, -25290):
        message = "macOS 登录钥匙串不可用；请在“钥匙串访问”中检查登录钥匙串，或配置 ARK_API_KEY 环境变量。"
        reason = "unavailable"
    elif status == -25292:
        message = "macOS 钥匙串只读，无法保存；请检查登录钥匙串权限，或配置 ARK_API_KEY 环境变量。"
        reason = "unavailable"
    else:
        message = "macOS 钥匙串操作失败；请检查登录钥匙串及系统授权，或配置 ARK_API_KEY 环境变量。"
        reason = "unavailable"
    return CredentialError(f"{message}（OSStatus {status}）", status=status, reason=reason)


def _wipe(buffer):
    if buffer:
        ctypes.memset(ctypes.addressof((ctypes.c_ubyte * len(buffer)).from_buffer(buffer)), 0, len(buffer))


class _NativeKeychain:
    """Small ctypes bridge; tests replace it and never access a real keychain.

    CF objects created/copied here are released on all paths.  The CFData used
    for saving borrows a mutable Python buffer until the native operation and
    dictionary release finish.  The caller then wipes that buffer.  Python's
    input/output strings cannot be reliably erased, so retain no extra copy.
    """

    def __init__(self):
        try:
            self.cf = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
            self.security = ctypes.CDLL("/System/Library/Frameworks/Security.framework/Security")
            pointer = ctypes.c_void_p
            index = ctypes.c_long
            signatures = {
                "CFStringCreateWithCString": ([pointer, ctypes.c_char_p, ctypes.c_uint32], pointer),
                "CFDataCreateWithBytesNoCopy": ([pointer, pointer, index, pointer], pointer),
                "CFDictionaryCreate": ([pointer, pointer, pointer, index, pointer, pointer], pointer),
                "CFDataGetLength": ([pointer], index),
                "CFDataGetBytePtr": ([pointer], pointer),
                "CFGetTypeID": ([pointer], ctypes.c_ulong),
                "CFDataGetTypeID": ([], ctypes.c_ulong),
                "CFRelease": ([pointer], None),
            }
            for name, (arguments, result) in signatures.items():
                function = getattr(self.cf, name)
                function.argtypes, function.restype = arguments, result
            for name, arguments in {
                "SecItemAdd": [pointer, ctypes.POINTER(pointer)],
                "SecItemUpdate": [pointer, pointer],
                "SecItemCopyMatching": [pointer, ctypes.POINTER(pointer)],
            }.items():
                function = getattr(self.security, name)
                function.argtypes, function.restype = arguments, ctypes.c_int32
            self._key_callbacks = ctypes.addressof(ctypes.c_char.in_dll(self.cf, "kCFTypeDictionaryKeyCallBacks"))
            self._value_callbacks = ctypes.addressof(ctypes.c_char.in_dll(self.cf, "kCFTypeDictionaryValueCallBacks"))
            # Check all symbols before any secret enters the bridge.
            for name in ("kSecClass", "kSecClassGenericPassword", "kSecAttrService", "kSecAttrAccount",
                         "kSecValueData", "kSecMatchLimit", "kSecMatchLimitOne", "kSecReturnData",
                         "kSecReturnAttributes", "kSecUseAuthenticationUI", "kSecUseAuthenticationUIFail"):
                self._constant(name)
            self._constant("kCFBooleanTrue", core=True)
            self._constant("kCFAllocatorNull", core=True)
        except (OSError, AttributeError, ValueError):
            raise CredentialError("无法加载 macOS 钥匙串服务；请检查系统，或配置 ARK_API_KEY 环境变量。") from None

    def _constant(self, name, *, core=False):
        result = ctypes.c_void_p.in_dll(self.cf if core else self.security, name).value
        if not result:
            raise ValueError("missing native constant")
        return result

    @contextmanager
    def _objects(self):
        objects = []
        try:
            yield objects
        finally:
            for item in reversed(objects):
                self.cf.CFRelease(item)

    @staticmethod
    def _remember(objects, item):
        if not item:
            raise CredentialError("macOS 钥匙串内存分配失败，请稍后重试。")
        objects.append(item)
        return item

    def _string(self, objects, text):
        return self._remember(objects, self.cf.CFStringCreateWithCString(None, text.encode("utf-8"), _UTF8))

    def _dictionary(self, objects, pairs):
        keys = (ctypes.c_void_p * len(pairs))(*(key for key, _ in pairs))
        values = (ctypes.c_void_p * len(pairs))(*(value for _, value in pairs))
        return self._remember(objects, self.cf.CFDictionaryCreate(
            None, keys, values, len(pairs), self._key_callbacks, self._value_callbacks))

    def _identity(self, objects):
        return [
            (self._constant("kSecClass"), self._constant("kSecClassGenericPassword")),
            (self._constant("kSecAttrService"), self._string(objects, KEYCHAIN_SERVICE)),
            (self._constant("kSecAttrAccount"), self._string(objects, KEYCHAIN_ACCOUNT)),
        ]

    def _data(self, objects, buffer):
        # CFData borrows this storage; _objects releases it before the caller
        # wipes it. kCFAllocatorNull prevents CF from freeing Python's memory.
        pointer = (ctypes.c_ubyte * len(buffer)).from_buffer(buffer)
        return self._remember(objects, self.cf.CFDataCreateWithBytesNoCopy(
            None, pointer, len(buffer), self._constant("kCFAllocatorNull", core=True)))

    def add(self, buffer):
        with self._objects() as objects:
            query = self._dictionary(objects, self._identity(objects) + [
                (self._constant("kSecValueData"), self._data(objects, buffer))])
            return int(self.security.SecItemAdd(query, None))

    def update(self, buffer):
        with self._objects() as objects:
            query = self._dictionary(objects, self._identity(objects))
            attributes = self._dictionary(objects, [
                (self._constant("kSecValueData"), self._data(objects, buffer))])
            return int(self.security.SecItemUpdate(query, attributes))

    def _copy(self, *, secret):
        with self._objects() as objects:
            query = self._dictionary(objects, self._identity(objects) + [
                (self._constant("kSecMatchLimit"), self._constant("kSecMatchLimitOne")),
                (self._constant("kSecUseAuthenticationUI"), self._constant("kSecUseAuthenticationUIFail")),
                (self._constant("kSecReturnData" if secret else "kSecReturnAttributes"),
                 self._constant("kCFBooleanTrue", core=True)),
            ])
            result = ctypes.c_void_p()
            status = int(self.security.SecItemCopyMatching(query, ctypes.byref(result)))
            if result.value:
                objects.append(result.value)
            if status != _SUCCESS or not secret:
                return status, None
            if not result.value or self.cf.CFGetTypeID(result) != self.cf.CFDataGetTypeID():
                raise CredentialError("钥匙串中的 API Key 数据无效；请运行 nolan ark-login 重新保存。")
            length = int(self.cf.CFDataGetLength(result))
            if length <= 0 or length > MAX_KEY_BYTES:
                raise CredentialError("钥匙串中的 API Key 数据无效；请运行 nolan ark-login 重新保存。")
            source = self.cf.CFDataGetBytePtr(result)
            if not source:
                raise CredentialError("无法读取钥匙串中的 API Key；请检查登录钥匙串。")
            buffer = bytearray(length)
            ctypes.memmove((ctypes.c_ubyte * length).from_buffer(buffer), source, length)
            return status, buffer

    def read(self):
        return self._copy(secret=True)

    def exists(self):
        status, _ = self._copy(secret=False)
        return status


def _get_bridge():
    if sys.platform != "darwin":
        raise CredentialError("仅 macOS 支持将 API Key 保存到钥匙串；此系统请配置 ARK_API_KEY 环境变量。")
    return _NativeKeychain()


def save_ark_key(key):
    """Explicitly add/update one Ark key in the current user's Keychain."""
    key = _validate_key(key)
    bridge = _get_bridge()
    buffer = bytearray(key, "utf-8")
    try:
        status = bridge.add(buffer)
        if status == _DUPLICATE_ITEM:
            status = bridge.update(buffer)
        if status != _SUCCESS:
            raise _error(status)
    finally:
        _wipe(buffer)


def load_ark_key():
    """Return the saved key internally, or None; never erase a denied item."""
    if sys.platform != "darwin":
        return None
    status, buffer = _get_bridge().read()
    try:
        if status == _ITEM_NOT_FOUND:
            return None
        if status != _SUCCESS:
            raise _error(status)
        if not isinstance(buffer, bytearray):
            raise CredentialError("钥匙串中的 API Key 数据无效；请运行 nolan ark-login 重新保存。")
        try:
            return _validate_key(buffer.decode("utf-8"))
        except (UnicodeError, CredentialError):
            raise CredentialError("钥匙串中的 API Key 数据无效；请运行 nolan ark-login 重新保存。") from None
    finally:
        if isinstance(buffer, bytearray):
            _wipe(buffer)


def _environment_key(env_name):
    if not isinstance(env_name, str) or not env_name or len(env_name) > 512 or any(
            ord(char) < 32 or ord(char) == 127 for char in env_name):
        raise CredentialError("API Key 环境变量名称无效，请检查 Nolan 配置。", reason="invalid")
    value = os.environ.get(env_name)
    if value is None or not value.strip():
        return None
    return _validate_key(value)


def resolve_ark_key(env_name=DEFAULT_ARK_ENV):
    """Environment wins; only the standard Ark variable falls back to Keychain."""
    value = _environment_key(env_name)
    if value is not None:
        return value
    return load_ark_key() if env_name == DEFAULT_ARK_ENV else None


def ark_key_status(env_name=DEFAULT_ARK_ENV):
    """Check actual noninteractive readability; return only safe status fields.

    Keychain attributes can be readable even when an item's password is denied.
    Use the same data-read path as production, discard its result immediately,
    and never expose the value through doctor/UI. No API/network call is made.
    """
    try:
        if _environment_key(env_name) is not None:
            return {"available": True, "source": "environment", "message": "已从环境变量配置火山方舟 API Key。"}
        if env_name != DEFAULT_ARK_ENV:
            return {"available": False, "source": "missing", "message": "指定的 API Key 环境变量尚未配置。"}
        if sys.platform != "darwin":
            raise CredentialError("仅 macOS 支持钥匙串；此系统请配置 ARK_API_KEY 环境变量。")
        if load_ark_key() is None:
            return {"available": False, "source": "missing", "message": "尚未保存火山方舟 API Key；请运行 nolan ark-login。"}
        return {"available": True, "source": "keychain", "message": "火山方舟 API Key 已保存在 macOS 钥匙串，并可在后台读取。"}
    except CredentialError as exc:
        return {"available": False, "source": "locked" if exc.reason == "locked" else "unavailable", "message": str(exc)}
