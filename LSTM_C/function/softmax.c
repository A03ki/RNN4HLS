#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/*
やること
ヘッダファイルにする

デバックが古典的だけど気にしない
*/

void softmax(float *, int);
int main(void){
    // Softmax関数
    float x[5] = {2.0, 4.0, 3.0, 6.0, 2.0};
    int t;
    softmax(x, 5);

    // softmax確認用
    int len = sizeof(x) / sizeof(x[0]);
    for(t=0;t<len;t++){
        printf("%f\n", x[t]);
    }

}

void softmax(float *input_x, int len){
    int i;
    // float confirmation_sum = 0; // 総和確認用
    float max = input_x[0];
    float sum = 0.0;

    // 配列の最大値を見つける
    for(i=1;i<len;i++){
        if(input_x[i] > max){
            max = input_x[i];
        }
    }

    for(i=0;i<len;i++){
        input_x[i] -= max;  // オーバーフロー対策
        printf("input_x[%d] - max = %f\n", i, input_x[i]);
        sum += expf(input_x[i]);
        printf("%d回目のsum: %f\n", i, sum);
    }

    for(i=0;i<len;i++){
        input_x[i] = expf(input_x[i]) / sum;
        // confirmation_sum += input_x[i];
    }

    // printf("総和(1になることの確認): %f\n", confirmation_sum);

}
