import types
import unittest

from style_shapes.training import _install_negative_trace_hook


class NegativeTraceHookTest(unittest.TestCase):
    def test_captures_inner_factorized_resolver_without_changing_result(self):
        owner = types.SimpleNamespace(_style_shapes_last_negatives=None)

        class Core:
            def _resolve_prototype_aware_negatives(self, marker=None):
                self.marker = marker
                return ([11, 12], [21, 22], {"kept": True})

        core = Core()
        _install_negative_trace_hook(owner, core)
        result = core._resolve_prototype_aware_negatives(marker="called")

        self.assertEqual(result, ([11, 12], [21, 22], {"kept": True}))
        self.assertEqual(owner._style_shapes_last_negatives, ([11, 12], [21, 22]))
        self.assertEqual(core.marker, "called")


if __name__ == "__main__":
    unittest.main()
