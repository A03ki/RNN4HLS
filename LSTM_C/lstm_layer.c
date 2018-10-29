#include <stdio.h>
#include <math.h>

/*
「今後考えるべきこと」
まだ行列積計算xWができない
再帰のところと並列処理のところをわけて作らないと並列処理できない
ht * sigmoid(W_o)のところはfor文を分けるべきか
sigmoidなどの関数を通すのも行列積計算(並列処理可能部)のときに行うべかか
mallocは極力使いたくないが, どうすればいいか(FPGAに使えるかが不明, 使えるなら使う)
*/

float sigmoid(float x){
    return (1 / (1 + exp(-x)));
}

void product_array(float array_c[][3], float array_W_f[][3], float array_W_i[][3], float array_W_c[][3], float array_W_o[][3], int row){
    int i,j;
    int k = 0;
    int l = 0;
    for (i=0;i<row;i++){
        for (j=0;j<3;j++){
            array_c[i][j] = sigmoid(array_W_f[i][j]) * array_c[k][l] + sigmoid(array_W_i[i][j]) * tanhf(array_W_c[i][j]);
            printf("%f\n", array_c[i][j]);
            array_c[i][j] = array_W_o[i][j] * array_c[i][j];
            printf("%f\n", array_c[i][j]);
            printf("(%d, %d), (%d, %d)\n", i, j, k, l);
            k = i;
            l = j;
        }
    }
}

int main(void){
    float c_t[3][3] = {{0.0}};
    float W_f[3][3] = {{1.0,2.0,3.0}, {1.0,2.0,3.0}, {1.0,2.0,3.0}};
    float W_i[3][3] = {{1.0,2.0,3.0}, {1.0,2.0,3.0}, {1.0,2.0,3.0}};
    float W_c[3][3] = {{1.0,2.0,3.0}, {1.0,2.0,3.0}, {1.0,2.0,3.0}};
    float W_o[3][3] = {{1.0,2.0,3.0}, {1.0,2.0,3.0}, {1.0,2.0,3.0}};
    int i,j;
    int row = 3;
    product_array(c_t, W_f, W_i, W_c, W_o, row);

    for (i=0;i<3;i++){
        for (j=0;j<3;j++){
            printf("%f ",c_t[i][j]);
        }
    }
    return 0;
}
