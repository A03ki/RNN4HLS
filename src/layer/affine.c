/**
 * @fn void affine(float *output, const float *input_x, const float *weight, const float *bais, int row, int matrix_k, int column)
 * @brief Affine変換をする関数
 * @param[out] output 出力用の一次元配列(行列積後の要素数と合わせる)
 * @param[in,out] input_x 重みと行列積を行う一次元配列
 * @param[in,out] weight 一次元配列からなる重み
 * @param[in,out] bais 一次元配列からなるバイアス
 * @param[in] row input_xの行の長さ
 * @param[in] matrix_k input_kの列の長さ
 * @param[in] column weightの列の長さ
 * @return void
 */

#include <stdio.h>
#include "affine.h"

void affine(float *output, const float *input_x, const float *weight, const float *bais, int row, int matrix_k, int column) {
        int i, j, k;

        // 初期化
        for(i=0; i<row*column; i++) {
                output[i] = 0.0;
        }

        // 行列積
        for (i = 0; i < row; i++) {
                for (k = 0; k < matrix_k; k++) {
                        for (j = 0; j < column; j++) {
                                output[i * column + j] += input_x[i * matrix_k + k] * weight[k * column + j];
                        }
                }
        }

        // バイアス加算
        for (i = 0; i < row; i++) {
                for (j = 0; j < column; j++) {
                        output[i * column + j] += bais[j];
                }
        }
}
