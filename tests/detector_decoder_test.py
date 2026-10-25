"""Guard the meaning and lifetime of decoder diagnostic evidence."""
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import reproduce_detector_rectangle as rectangle


class DecoderDiagnostic(unittest.TestCase):
    def test_consumed_references_use_the_original_coordinate_bound(self):
        native = np.zeros((1, 300, 1, 4), np.float32)
        actual = native.copy(); actual[0, 0, 0, 0] = 0.0001
        try:
            result = rectangle.compare_tensor(actual, native, 'layer0.reference_points')
        except ValueError as error:
            self.fail(f'Decoder coordinate comparison is unavailable: {error}')
        self.assertFalse(result['passedAtOriginalBound'])
        self.assertEqual(result['failedElements'], 1)
        self.assertEqual(result['atol'], 2e-5)
        self.assertEqual(result['rtol'], 1e-4)

    def test_decoder_comparison_rejects_wrong_shape_dtype_or_nonfinite(self):
        native = np.zeros((1, 300, 128), np.float32)
        for bad in (native[:, :1], native.astype(np.float64), np.full_like(native, np.nan)):
            with self.subTest(shape=bad.shape, dtype=bad.dtype), self.assertRaises(ValueError):
                rectangle.compare_tensor(bad, native, 'layer2.output')

    def test_same_input_replay_consumes_every_ort_operand(self):
        import diagnose_detector_decoder as decoder
        class Layer(torch.nn.Module):
            def forward(self, *, hidden_states, position_embeddings, reference_points,
                        encoder_hidden_states, spatial_shapes, spatial_shapes_list,
                        encoder_attention_mask, output_attentions):
                if spatial_shapes.tolist() != [[40, 40], [20, 20]] or spatial_shapes_list != [(40, 40), (20, 20)]:
                    raise ValueError('Wrong spatial operands')
                if encoder_attention_mask is not None or output_attentions is not False:
                    raise ValueError('Wrong eval path')
                return (hidden_states + 2 * position_embeddings + 3 * reference_points.mean()
                        + 4 * encoder_hidden_states.mean(),)
        operands = {
            'hidden_states': np.full((1, 300, 128), 1, np.float32),
            'position_embeddings': np.full((1, 300, 128), 2, np.float32),
            'reference_points': np.full((1, 300, 1, 4), 3, np.float32),
            'encoder_hidden_states': np.full((1, 2000, 128), 4, np.float32),
        }
        result = decoder.replay_layer(Layer(), operands, np.array([[40, 40], [20, 20]], np.int64))
        np.testing.assert_array_equal(result, np.full((1, 300, 128), 30, np.float32))

    def test_capture_owns_arrays_and_is_removed_after_an_exception(self):
        import diagnose_detector_decoder as decoder
        linear = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            linear.weight.copy_(torch.eye(2))
        captured = {}
        source = torch.tensor([[1., 2.]])
        with self.assertRaisesRegex(RuntimeError, 'stop'):
            with decoder.capture_modules({'head': linear}, {}, captured):
                result = linear(source)
                source.add_(10)
                result.detach().add_(20)
                raise RuntimeError('stop')
        linear(torch.tensor([[7., 8.]]))
        np.testing.assert_array_equal(captured['head.input'], [[1., 2.]])
        np.testing.assert_array_equal(captured['head.output'], [[1., 2.]])

    def test_replay_rejects_invalid_spatial_operands(self):
        import diagnose_detector_decoder as decoder
        for shape in (np.array([[40, 40], [20, 20]], np.float32),
                      np.array([[20, 20], [40, 40]], np.int64)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                decoder.replay_layer(torch.nn.Identity(), {}, shape)

    def test_query_order_check_uses_the_consumed_embeddings_without_sorting(self):
        import diagnose_detector_decoder as decoder
        memory = np.arange(2000 * 128, dtype=np.float32).reshape(1, 2000, 128)
        indices = np.arange(300, dtype=np.int64)[None]
        consumed = memory[:, :300].copy()
        self.assertTrue(decoder.consumed_query_order(memory, consumed, indices)['matchesConsumedEmbeddings'])
        reversed_order = indices[:, ::-1].copy()
        self.assertFalse(decoder.consumed_query_order(memory, consumed, reversed_order)['matchesConsumedEmbeddings'])
        for bad in (indices.astype(np.int32), np.full((1, 300), 2000, np.int64),
                    np.zeros((1, 300), np.int64)):
            with self.subTest(dtype=bad.dtype), self.assertRaises(ValueError):
                decoder.consumed_query_order(memory, consumed, bad)


if __name__ == '__main__':
    unittest.main()
