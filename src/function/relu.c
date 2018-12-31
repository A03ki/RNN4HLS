/**
 * @fn void relu(float *input_x, int len)
 * @brief 1次元配列の各要素に対してReLUを通す関数
 * @param[out] input_x 一次元配列
 * @param[in] len 一次元配列の要素数
 * @return void
 * @detail 入力値input_xを書き換えてしまいます
 */

#include <stdio.h>
#include "relu.h"

void relu(float *input_x, 　int len) {
        int i;
        for (i = 0; i < len; i++) {
                if (input_x[i] < 0) {
                        input_x[i] = 0;
                }
                printf("%f\n", input_x[i]);
        }
}
