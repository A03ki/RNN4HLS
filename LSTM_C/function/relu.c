#include <stdio.h>

/*
一次元配列をReLUに通す

やること
ヘッダファイル作る
*/

void relu(float*, int);

int main(void){
    // ReLU
    float array[2*3] = {1.0, 3.3, -4.0, -2.1, 3.0, 0.0};
    int len = sizeof(array) / sizeof(array[0]);
    relu(array, len);
}

void relu(float *input_x, int len){
    int i;
    for (i = 0; i < len; i++){
        if (input_x[i] < 0){
            input_x[i] = 0;
        }
        printf("%f\n", input_x[i]);
    }
}
