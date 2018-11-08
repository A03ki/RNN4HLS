#include <stdio.h>

/*
やること
ヘッダファイルにする

推論時はsoftmaxは必要ないのでその代わりに最大値のindexを取得する関数
*/

int softmax(float *, int);
int main(void){
    // Softmax関数
    float x[5] = {2.0, 4.0, 3.0, 6.0, 2.0};
    int class_idx = softmax(x, 5);

    // 動作確認用
    printf("max_index: %d\n", class_idx);

}

int softmax(float *input_x, int len){
    int i;
    int max_index = 0;

    // 配列の最大値のインデックスを見つける
    for(i=1;i<len;i++){
        if(input_x[i] > input_x[max_index]){
            max_index = i;
        }
    }
    return max_index;

}
