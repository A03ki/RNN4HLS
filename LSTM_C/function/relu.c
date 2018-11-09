#include <stdio.h>
#include "relu.h"

/*
*input_x : 一次元配列のポインタ変数
len : 一次元配列の長さ
*/

void relu(float *input_x, int len){
    int i;
    for (i = 0; i < len; i++){
        if (input_x[i] < 0){
            input_x[i] = 0;
        }
        printf("%f\n", input_x[i]);}
}
