import cffi
import numpy as np
import unittest

from make import run_make


class TestAffine(unittest.TestCase):
    def test_affine(self):
        stdout, stderr = run_make()
        print(stdout, stderr)
        ffi = cffi.FFI()
        ffi.cdef("""
        void affine(float *output, const float *input_x,
        const float *weight, const float *bais,
        int row, int matrix_k, int column);
        """)
        lib = ffi.dlopen("build/libsinf.so")
        for _ in range(100):
            row, matrix_k, column = np.random.randint(30, 100, size=(3))
            input_x = np.random.randn(row, matrix_k).astype('f')
            weight = np.random.randn(matrix_k, column).astype('f')
            bias = np.random.randn(column).astype('f')
            expected = np.dot(input_x, weight) + bias
            actual = np.zeros_like(expected.flatten())  # 初期化

            p_input_x = ffi.cast("float *", input_x.flatten().ctypes.data)
            p_weight = ffi.cast("float *", weight.flatten().ctypes.data)
            p_output = ffi.cast("float *", actual.ctypes.data)
            p_bias = ffi.cast("float *", bias.ctypes.data)
            lib.affine(p_output, p_input_x, p_weight, p_bias,
                       row, matrix_k, column)
            actual = actual.reshape(row, column)
            print((actual - expected))
            np.testing.assert_allclose(actual, expected, atol=1e-04)


if __name__ == "__main__":
    unittest.main()
