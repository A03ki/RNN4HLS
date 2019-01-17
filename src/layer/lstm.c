/**
 * @fn void lstm(float *output, const float *input_x, const float *weight_x, const float *weight_h, const float *bias, int row, int matrix_k, int column)
 * @brief 一次元配列を一方向LSTMレイヤに通す関数
 * @param[out] output 出力用の一次元配列(行列積後の要素数と合わせる)
 * @param[in,out] input_x 関数内の重みと行列積を行う一次元配列
 * @param[in,out] weight_x input_xと行列積を行う一次元配列の重み
 * @param[in,out] weight_h 時刻t-1の時のoutputと行列積を行う一次元配列の重み
 * @param[in,out] bias 一次元配列のバイアス. 転置した重みの行と同じ長さを持つ
 * @param[in] row input_xが二次元配列の時の列の長さ
 * @param[in] matrix_k input_xが二次元配列の時の行の長さ
 * @param[in] column weightが二次元配列の時の列の長さ
 * @return void
 * @detail Pytorchの重みは転置, バイアスは2つを足して使う
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../function/sigmoid.h"
#include "lstm.h"

/*
   outputの初期化
 */

void lstm(float *output, const float *input_x, const float *weight_x, const float *weight_h, const float *bias, int row, int matrix_k, int column) {
        int i, j, k, t, l, t_pre, t_element;
        t_pre = 0;
        int ifgo_t_size = column * 4; // tのときのifgoのsize

        // 初期化
        float array_fgio[4000] = {0};
        float array_c[10]      = {0};
        for(i=0; i<row*column; i++) {
                output[i] = 0.0;
        }

        // アフィン変換
        for (i = 0; i < row; i++) {
                for (k = 0; k < matrix_k; k++) {
                        for (j = 0; j < ifgo_t_size; j++) {
                                array_fgio[i * ifgo_t_size + j] += input_x[i * matrix_k + k] * weight_x[k * column + j];
                        }
                }
        }
        for (i = 0; i < row; i++) {
                for (j = 0; j < ifgo_t_size; j++) {
                        array_fgio[i * ifgo_t_size + j] += bias[j];
                }
        }

        for (t = 0; t < row; t++) {
                for (i = 0; i < column; i++) {
                        for (j = 0; j < ifgo_t_size; j++) {
                                array_fgio[t * ifgo_t_size + j] += output[t_pre + i] * weight_h[i * ifgo_t_size + j];
                        }
                }
                for (l = 0; l < column; l++) {
                        t_element                          = t * ifgo_t_size + l;
                        array_fgio[t_element]              = sigmoid(array_fgio[t_element]);
                        array_fgio[column + t_element]     = sigmoid(array_fgio[column + t_element]);
                        array_fgio[2 * column + t_element] = tanhf(array_fgio[2 * column + t_element]);
                        array_fgio[3 * column + t_element] = sigmoid(array_fgio[3 * column + t_element]);
                        array_c[l]                         = array_fgio[column + t_element] * array_c[l] + array_fgio[t_element] * array_fgio[2 * column + t_element];
                        output[t * column + l]             = array_fgio[3 * column + t_element] * tanhf(array_c[l]);
                }
                t_pre = t * column; // t-1の役割
        }
}
