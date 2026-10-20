"""Synthetic credentials only; native Keychain and all subprocesses are mocked."""

import ctypes
import io
import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from norlan import credentials
from norlan.credentials import CredentialError


SYNTHETIC_KEY = "synthetic-ark-unit-test-only"


class CredentialTests(unittest.TestCase):
    def setUp(self):
        self.platform = patch.object(credentials.sys, "platform", "darwin")
        self.platform.start()
        self.addCleanup(self.platform.stop)
        environment = patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        self.bridge = Mock()
        self.bridge.add.return_value = 0
        self.bridge.update.return_value = 0
        self.bridge.exists.return_value = -25300
        self.bridge.read.return_value = (-25300, None)
        self.factory = patch.object(credentials, "_get_bridge", return_value=self.bridge)
        self.factory.start()
        self.addCleanup(self.factory.stop)
        # A credential must never become an argument, stdin payload or a file.
        for name in ("subprocess.run", "subprocess.Popen", "subprocess.check_output", "builtins.open"):
            forbidden = patch(name, side_effect=AssertionError("external access forbidden"))
            forbidden.start()
            self.addCleanup(forbidden.stop)

    def test_add_uses_memory_then_wipes_it_without_stdout_or_stderr(self):
        seen = []
        self.bridge.add.side_effect = lambda data: seen.append(bytes(data)) or 0
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch("sys.stdout", stdout), patch("sys.stderr", stderr):
            credentials.save_ark_key("  " + SYNTHETIC_KEY + "  ")
        self.assertEqual(seen, [SYNTHETIC_KEY.encode()])
        self.assertEqual(self.bridge.add.call_args.args[0], bytearray(len(SYNTHETIC_KEY)))
        self.bridge.update.assert_not_called()
        self.assertEqual(stdout.getvalue() + stderr.getvalue(), "")

    def test_duplicate_is_updated_once_without_delete_or_clear(self):
        self.bridge.add.return_value = -25299
        seen = []
        self.bridge.update.side_effect = lambda data: seen.append(bytes(data)) or 0
        credentials.save_ark_key(SYNTHETIC_KEY)
        self.assertEqual(seen, [SYNTHETIC_KEY.encode()])
        self.bridge.update.assert_called_once()
        self.assertEqual(self.bridge.update.call_args.args[0], bytearray(len(SYNTHETIC_KEY)))
        self.bridge.delete.assert_not_called()

    def test_save_denial_keeps_existing_item_and_wipes_buffer(self):
        self.bridge.add.return_value = -25308
        with self.assertRaisesRegex(CredentialError, "解锁") as caught:
            credentials.save_ark_key(SYNTHETIC_KEY)
        self.assertEqual(caught.exception.status, -25308)
        self.assertNotIn(SYNTHETIC_KEY, str(caught.exception))
        self.bridge.update.assert_not_called()
        self.bridge.delete.assert_not_called()
        self.assertEqual(self.bridge.add.call_args.args[0], bytearray(len(SYNTHETIC_KEY)))

    def test_update_denial_is_not_retried_or_replaced(self):
        self.bridge.add.return_value = -25299
        self.bridge.update.return_value = -25293
        with self.assertRaisesRegex(CredentialError, "授权"):
            credentials.save_ark_key(SYNTHETIC_KEY)
        self.assertEqual(self.bridge.add.call_count, 1)
        self.assertEqual(self.bridge.update.call_count, 1)
        self.bridge.delete.assert_not_called()

    def test_invalid_values_are_rejected_before_native_access(self):
        for value in (None, 12, "", "   ", SYNTHETIC_KEY + "\n", "a\rb", "a\0b", "a\tb",
                      "a\x7fb", "x" * (credentials.MAX_KEY_BYTES + 1), "\ud800"):
            with self.subTest(value_type=type(value).__name__):
                with self.assertRaises(CredentialError) as caught:
                    credentials.save_ark_key(value)
                self.assertNotIn(SYNTHETIC_KEY, str(caught.exception))
        self.bridge.add.assert_not_called()
        self.bridge.update.assert_not_called()

    def test_key_format_is_not_bound_to_vendor_prefix_or_ascii(self):
        credentials.save_ark_key("synthetic-中文-key")
        self.bridge.add.assert_called_once()

    def test_load_returns_key_to_internal_caller_and_wipes_native_copy(self):
        buffer = bytearray(SYNTHETIC_KEY, "utf-8")
        self.bridge.read.return_value = (0, buffer)
        self.assertEqual(credentials.load_ark_key(), SYNTHETIC_KEY)
        self.assertEqual(buffer, bytearray(len(SYNTHETIC_KEY)))
        self.bridge.exists.assert_not_called()

    def test_missing_is_none_and_is_not_cached(self):
        self.assertIsNone(credentials.load_ark_key())
        self.bridge.read.return_value = (0, bytearray(SYNTHETIC_KEY, "utf-8"))
        self.assertEqual(credentials.load_ark_key(), SYNTHETIC_KEY)
        self.assertEqual(self.bridge.read.call_count, 2)

    def test_successful_reads_are_not_cached_across_rotation(self):
        self.bridge.read.side_effect = [(0, bytearray(SYNTHETIC_KEY, "utf-8")),
                                        (0, bytearray("synthetic-rotated-key", "utf-8"))]
        self.assertEqual(credentials.load_ark_key(), SYNTHETIC_KEY)
        self.assertEqual(credentials.load_ark_key(), "synthetic-rotated-key")

    def test_load_locked_is_actionable_and_never_treated_as_missing(self):
        for status in (-25308, -25315, -25293, -128):
            with self.subTest(status=status):
                self.bridge.read.return_value = (status, None)
                with self.assertRaises(CredentialError) as caught:
                    credentials.load_ark_key()
                self.assertEqual(caught.exception.reason, "locked")
                self.assertIn("钥匙串", str(caught.exception))
                self.assertNotIn(SYNTHETIC_KEY, str(caught.exception))
        self.bridge.add.assert_not_called()
        self.bridge.update.assert_not_called()

    def test_corrupt_stored_values_are_rejected_without_disclosure(self):
        for value in (b"", b"\xff", SYNTHETIC_KEY.encode() + b"\n", b"x" * (credentials.MAX_KEY_BYTES + 1)):
            buffer = bytearray(value)
            self.bridge.read.return_value = (0, buffer)
            with self.subTest(size=len(value)), self.assertRaises(CredentialError) as caught:
                credentials.load_ark_key()
            self.assertNotIn(SYNTHETIC_KEY, str(caught.exception))
            self.assertEqual(buffer, bytearray(len(value)))

    def test_environment_has_priority_and_does_not_touch_keychain(self):
        with patch.dict(os.environ, {"ARK_API_KEY": "  " + SYNTHETIC_KEY + "  "}):
            self.assertEqual(credentials.resolve_ark_key(), SYNTHETIC_KEY)
            status = credentials.ark_key_status()
        self.assertEqual(status["source"], "environment")
        self.assertTrue(status["available"])
        self.assertNotIn(SYNTHETIC_KEY, repr(status))
        self.bridge.read.assert_not_called()
        self.bridge.exists.assert_not_called()

    def test_invalid_environment_does_not_silently_fall_back(self):
        with patch.dict(os.environ, {"ARK_API_KEY": SYNTHETIC_KEY + "\n"}):
            with self.assertRaises(CredentialError):
                credentials.resolve_ark_key()
            status = credentials.ark_key_status()
        self.assertEqual(status["source"], "unavailable")
        self.assertFalse(status["available"])
        self.assertNotIn(SYNTHETIC_KEY, repr(status))
        self.bridge.read.assert_not_called()
        self.bridge.exists.assert_not_called()

    def test_blank_environment_falls_back_but_custom_names_never_do(self):
        self.bridge.read.return_value = (0, bytearray(SYNTHETIC_KEY, "utf-8"))
        with patch.dict(os.environ, {"ARK_API_KEY": "   "}):
            self.assertEqual(credentials.resolve_ark_key(), SYNTHETIC_KEY)
        self.bridge.read.reset_mock()
        self.assertIsNone(credentials.resolve_ark_key("SYNTHETIC_CUSTOM_KEY"))
        self.assertEqual(credentials.ark_key_status("SYNTHETIC_CUSTOM_KEY")["source"], "missing")
        self.bridge.read.assert_not_called()
        self.bridge.exists.assert_not_called()
        with patch.dict(os.environ, {"SYNTHETIC_CUSTOM_KEY": SYNTHETIC_KEY}):
            self.assertEqual(credentials.resolve_ark_key("SYNTHETIC_CUSTOM_KEY"), SYNTHETIC_KEY)

    def test_status_checks_data_readability_but_never_returns_secret(self):
        buffer = bytearray(SYNTHETIC_KEY, "utf-8")
        self.bridge.read.return_value = (0, buffer)
        self.bridge.exists.return_value = 0
        status = credentials.ark_key_status()
        self.assertEqual(set(status), {"available", "source", "message"})
        self.assertEqual(status["source"], "keychain")
        self.assertTrue(status["available"])
        self.assertNotIn(SYNTHETIC_KEY, repr(status))
        self.bridge.read.assert_called_once()
        self.bridge.exists.assert_not_called()
        self.assertEqual(buffer, bytearray(len(SYNTHETIC_KEY)))

    def test_readable_attributes_cannot_mask_denied_password(self):
        self.bridge.exists.return_value = 0
        self.bridge.read.return_value = (-25308, None)
        status = credentials.ark_key_status()
        self.assertFalse(status["available"])
        self.assertEqual(status["source"], "locked")
        self.bridge.exists.assert_not_called()
        self.bridge.read.assert_called_once()

    def test_status_rechecks_data_on_each_call_after_rotation_or_lock(self):
        self.bridge.read.side_effect = [(0, bytearray(SYNTHETIC_KEY, "utf-8")), (-25308, None)]
        self.assertTrue(credentials.ark_key_status()["available"])
        self.assertEqual(credentials.ark_key_status()["source"], "locked")
        self.assertEqual(self.bridge.read.call_count, 2)

    def test_missing_locked_and_unavailable_statuses_are_distinct(self):
        for native_status, source in ((-25300, "missing"), (-25308, "locked"), (-25293, "locked"),
                                      (-25291, "unavailable"), (-50, "unavailable")):
            with self.subTest(native_status=native_status):
                self.bridge.read.return_value = (native_status, None)
                status = credentials.ark_key_status()
                self.assertFalse(status["available"])
                self.assertEqual(status["source"], source)
                self.assertNotIn(SYNTHETIC_KEY, repr(status))
        self.assertEqual(self.bridge.read.call_count, 5)
        self.bridge.exists.assert_not_called()

    def test_non_macos_environment_still_works_and_save_is_explicit_error(self):
        self.factory.stop()
        with patch.object(credentials.sys, "platform", "linux"):
            self.assertIsNone(credentials.load_ark_key())
            self.assertIsNone(credentials.resolve_ark_key())
            self.assertEqual(credentials.ark_key_status()["source"], "unavailable")
            with self.assertRaisesRegex(CredentialError, "仅 macOS"):
                credentials.save_ark_key(SYNTHETIC_KEY)
            with patch.dict(os.environ, {"ARK_API_KEY": SYNTHETIC_KEY}):
                self.assertEqual(credentials.resolve_ark_key(), SYNTHETIC_KEY)
                self.assertEqual(credentials.ark_key_status()["source"], "environment")


class NativeBridgeTests(unittest.TestCase):
    """Exercise native dictionary/lifetime mechanics using synthetic pointers."""

    def setUp(self):
        self.bridge = object.__new__(credentials._NativeKeychain)
        self.bridge.cf = SimpleNamespace(CFRelease=Mock())
        self.bridge.security = SimpleNamespace(SecItemCopyMatching=Mock(), SecItemAdd=Mock(return_value=0),
                                               SecItemUpdate=Mock(return_value=0))
        self.symbols = {}

        def symbol(name, **_):
            if name not in self.symbols:
                self.symbols[name] = len(self.symbols) + 100
            return self.symbols[name]

        self.bridge._constant = symbol
        self.bridge._string = lambda objects, text: self.bridge._remember(objects, 10 if text == credentials.KEYCHAIN_SERVICE else 11)
        self.pairs = []

        def dictionary(objects, pairs):
            self.pairs.append(pairs)
            return self.bridge._remember(objects, 500 + len(self.pairs))

        self.bridge._dictionary = dictionary

    def _copy_result(self, status, result):
        def copy(_, output):
            ctypes.cast(output, ctypes.POINTER(ctypes.c_void_p))[0] = result
            return status
        self.bridge.security.SecItemCopyMatching.side_effect = copy

    def test_metadata_query_is_noninteractive_and_never_asks_for_data(self):
        self._copy_result(0, 800)
        self.assertEqual(self.bridge.exists(), 0)
        pairs = dict(self.pairs[0])
        self.assertIn(self.symbols["kSecReturnAttributes"], pairs)
        self.assertNotIn("kSecReturnData", self.symbols)
        self.assertEqual(pairs[self.symbols["kSecUseAuthenticationUI"]], self.symbols["kSecUseAuthenticationUIFail"])
        self.assertEqual(self.bridge.cf.CFRelease.call_args_list[0].args, (800,))
        self.assertEqual(self.bridge.cf.CFRelease.call_count, 4)

    def test_read_copies_data_before_releasing_native_memory(self):
        synthetic = ctypes.create_string_buffer(SYNTHETIC_KEY.encode())
        self._copy_result(0, 800)
        self.bridge.cf.CFGetTypeID = Mock(return_value=7)
        self.bridge.cf.CFDataGetTypeID = Mock(return_value=7)
        self.bridge.cf.CFDataGetLength = Mock(return_value=len(SYNTHETIC_KEY))
        self.bridge.cf.CFDataGetBytePtr = Mock(return_value=ctypes.addressof(synthetic))
        status, buffer = self.bridge.read()
        self.assertEqual(status, 0)
        self.assertEqual(buffer, bytearray(SYNTHETIC_KEY, "utf-8"))
        self.assertIn(self.symbols["kSecReturnData"], dict(self.pairs[0]))
        self.bridge.cf.CFRelease.assert_any_call(800)
        credentials._wipe(buffer)

    def test_copy_error_still_releases_returned_object(self):
        self._copy_result(-25308, 800)
        self.assertEqual(self.bridge.read(), (-25308, None))
        self.bridge.cf.CFRelease.assert_any_call(800)
        self.assertEqual(self.bridge.cf.CFRelease.call_count, 4)

    def test_invalid_native_data_releases_objects_on_exception(self):
        self._copy_result(0, 800)
        self.bridge.cf.CFGetTypeID = Mock(return_value=8)
        self.bridge.cf.CFDataGetTypeID = Mock(return_value=7)
        with self.assertRaisesRegex(CredentialError, "数据无效"):
            self.bridge.read()
        self.bridge.cf.CFRelease.assert_any_call(800)
        self.assertEqual(self.bridge.cf.CFRelease.call_count, 4)

    def test_add_owns_borrowed_data_until_dictionary_is_released(self):
        buffer = bytearray(SYNTHETIC_KEY, "utf-8")
        self.bridge.cf.CFDataCreateWithBytesNoCopy = Mock(return_value=700)
        self.assertEqual(self.bridge.add(buffer), 0)
        create = self.bridge.cf.CFDataCreateWithBytesNoCopy.call_args.args
        self.assertEqual(ctypes.addressof(create[1]), ctypes.addressof((ctypes.c_ubyte * len(buffer)).from_buffer(buffer)))
        self.assertEqual(create[2], len(buffer))
        self.assertEqual(create[3], self.symbols["kCFAllocatorNull"])
        released = [call.args[0] for call in self.bridge.cf.CFRelease.call_args_list]
        self.assertEqual(released[:2], [501, 700])
        self.assertEqual(buffer, bytearray(SYNTHETIC_KEY, "utf-8"))
        credentials._wipe(buffer)

    def test_initializer_uses_frameworks_only_and_signed_status(self):
        cf, security = Mock(), Mock()
        with patch.object(credentials.ctypes, "CDLL", side_effect=[cf, security]) as loader, \
                patch.object(credentials.ctypes.c_char, "in_dll", return_value=ctypes.c_char()), \
                patch.object(credentials.ctypes.c_void_p, "in_dll", return_value=ctypes.c_void_p(200)):
            credentials._NativeKeychain()
        self.assertEqual([call.args[0] for call in loader.call_args_list], [
            "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation",
            "/System/Library/Frameworks/Security.framework/Security",
        ])
        self.assertIs(security.SecItemCopyMatching.restype, ctypes.c_int32)
        self.assertEqual(ctypes.sizeof(security.SecItemCopyMatching.restype), 4)

    def test_framework_load_failure_does_not_expose_arbitrary_exception_text(self):
        with patch.object(credentials.ctypes, "CDLL", side_effect=OSError(SYNTHETIC_KEY)):
            with self.assertRaises(CredentialError) as caught:
                credentials._NativeKeychain()
        self.assertNotIn(SYNTHETIC_KEY, str(caught.exception))


if __name__ == "__main__":
    unittest.main()
