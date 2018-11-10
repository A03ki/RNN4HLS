#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../function/sigmoid.h"

/*
sigmoid関数を分離させた
今後やること
affine関数をヘッダファイルに分ける
記憶セルをだだのキャッシュにする(余分に保持しない)
*/

void affine(float*, float*, float*, float*, int, int, int);
void lstm(float*, float*, int);

int main(void){
    int M, O, t;
    M=3, O=2;
    float *output_x;
    output_x = malloc(sizeof(float) * M * O);
    // 初期化
    for(t=0;t<(M*O);t++){
        output_x[t] = 0;
    }
    float input_x[3*2] = {0.1,0.2,0.3,0.4,0.5,0.6};
    lstm(output_x, input_x, M);
    // メモリ解放
    free(output_x);
}

void lstm(float *output, float *input_x, int row){
    int M, N, O, t, l,m,i,j,k;
    M=row, N = 2, O=2, m=0;

    float *array_f;
    float *array_g;
    float *array_i;
    float *array_o;
    float *array_c;
    array_f = malloc(sizeof(float) * M * O);
    array_g = malloc(sizeof(float) * M * O);
    array_i = malloc(sizeof(float) * M * O);
    array_o = malloc(sizeof(float) * M * O);
    array_c = malloc(sizeof(float) * M * O);

    // 初期化
    for(t=0;t<(M*O);t++){
        array_f[t] = 0;
        array_g[t] = 0;
        array_i[t] = 0;
        array_o[t] = 0;
        array_c[t] = 0;
    }
    // xの重み定義
    float weight_f_x[2*2] = {1.0, 2.0, 1.0, 2.0};
    float weight_g_x[2*2] = {1.0, 2.0, 1.0, 2.0};
    float weight_i_x[2*2] = {1.0, 2.0, 1.0, 2.0};
    float weight_o_x[2*2] = {1.0, 2.0, 1.0, 2.0};
    // hの重み定義
    float weight_f_h[2*2] = {1.0, 2.0, 1.0, 2.0};
    float weight_g_h[2*2] = {1.0, 2.0, 1.0, 2.0};
    float weight_i_h[2*2] = {1.0, 2.0, 1.0, 2.0};
    float weight_o_h[2*2] = {1.0, 2.0, 1.0, 2.0};
    // バイアス定義
    float bais_f[2] = {0.2, 0.3};
    float bais_g[2] = {0.2, 0.3};
    float bais_i[2] = {0.2, 0.3};
    float bais_o[2] = {0.2, 0.3};
    // アフィン変換
    affine(array_f, input_x, weight_f_x, bais_f, M, N, O);
    affine(array_g, input_x, weight_g_x, bais_g, M, N, O);
    affine(array_i, input_x, weight_i_x, bais_i, M, N, O);
    affine(array_o, input_x, weight_o_x, bais_o, M, N, O);


    /*
    // 行列積の確認
    for(t=0;t<(M*O);t++){
        printf("array_f[%d]=%f\n", t, array_f[t]);
        printf("array_g[%d]=%f\n", t, array_g[t]);
        printf("array_i[%d]=%f\n", t, array_i[t]);
        printf("array_o[%d]=%f\n", t, array_o[t]);
    }
    */


    for(i=0;i<M;i++){
        for(k=0;k<N;k++){
            for(j=0;j<O;j++){
                printf("(m+k, i*O+j)=(%d, %d)\n", m+k, i*O+j);
                //printf("output1[%d]=%f\n, weight_f_h[%d]=%f\n, ", m+j, output[m+j], k*O+j, weight_f_h[k*O+j]);
                printf("array_f[%d]=%f\n",i*O+j, array_f[i*O+j]);
                printf("weight_f_h=%f\n", weight_f_h[k*O+j]);
                printf("output[%d]=%f\n", m+k, output[m+k]);
                array_f[i*O+j] += output[m+k] * weight_f_h[k*O+j];
                array_g[i*O+j] += output[m+k] * weight_g_h[k*O+j];
                array_i[i*O+j] += output[m+k] * weight_i_h[k*O+j];
                array_o[i*O+j] += output[m+k] * weight_o_h[k*O+j];
                printf("array_f[%d]=%f\n",i*O+j, array_f[i*O+j]);
                //printf("array_g[%d]=%f\n",i*O+j, array_g[i*O+j]);
                //printf("array_i[%d]=%f\n",i*O+j, array_i[i*O+j]);
                //printf("array_o[%d]=%f\n",i*O+j, array_o[i*O+j]);
            }
        }
        for(l=0;l<O;l++){
            //printf("array_f1[%d]=%f\n",i*O+l, array_f[i*O+l]);
            //printf("array_g[%d]=%f\n",i*O+l, array_g[i*O+l]);
            //printf("array_i[%d]=%f\n",i*O+l, array_i[i*O+l]);
            //printf("array_o[%d]=%f\n",i*O+l, array_o[i*O+l]);
            array_f[i*O+l] = sigmoid(array_f[i*O+l]);
            //printf("array_f2[%d]=%f\n",i*O+l, array_f[i*O+l]);

            array_g[i*O+l] = tanhf(array_g[i*O+l]);
            array_i[i*O+l] = sigmoid(array_i[i*O+l]);
            array_o[i*O+l] = sigmoid(array_o[i*O+l]);
            array_c[i*O+l] = array_f[i*O+l] * array_c[m+l] + array_g[i*O+l] * array_i[i*O+l];
            output[i*O+l] = array_o[i*O+l] * tanhf(array_c[i*O+l]);

            //printf("array_f[%d]=%f\n",i*O+l, array_f[i*O+l]);
            //printf("array_g[%d]=%f\n",i*O+l, array_g[i*O+l]);
            //printf("array_i[%d]=%f\n",i*O+l, array_i[i*O+l]);
            //printf("array_o[%d]=%f\n",i*O+l, array_o[i*O+l]);
            //printf("array_c[%d]=%f\n",i*O+l, array_c[i*O+l]);
            //printf("output2[%d]=%f\n",i*O+l, output[i*O+l]);
        }
        m = i*O;  // t-1の役割
    }

    for(i=0;i<M;i++){
        for(j=0;j<O;j++){
            //printf("array_f[%d]=%f\n",i*O+j, array_f[i*O+j]);
            //printf("array_g[%d]=%f\n",i*O+j, array_g[i*O+j]);
            //printf("array_i[%d]=%f\n",i*O+j, array_i[i*O+j]);
            //printf("array_o[%d]=%f\n",i*O+j, array_o[i*O+j]);
            //printf("array_c[%d]=%f\n",i*O+j, array_c[j]);
            printf("output[%d]=%f\n",i*O+j, output[i*O+j]);
        }
    }

    // メモリ解放
    free(array_f);
    free(array_g);
    free(array_i);
    free(array_o);
}

void affine(float *output, float *input_x,  float *weight, float *bais, int row, int matrix_k, int column){
    /*
    for(i=0;i<row;i++){
        for(j=0;j<N;j++){
            printf("(i, j)=(%d, %d) ", i, j);
            printf("bar[%d]=%d\n",i*N+j, bar[i*N+j]);
        }
    }
    */
    int i, j, k;

    // 行列積
    for(i=0;i<row;i++){
        for(k=0;k<matrix_k;k++){
            for(j=0;j<column;j++){
            output[i*column+j] += input_x[i*matrix_k+k] * weight[k*column+j];
            //printf("(i, j, k)=(%d, %d, %d) ", i, j, k);
            //printf("output[%d]=%f\n",i*O+j, output[i*O+j]);
            }
        }
    }

    // バイアス加算
    for(i=0;i<row;i++){
            for(j=0;j<column;j++){
            //printf("output[%d]=%f\n",i*column+j, output[i*column+j]);
            //output[i*O+j] = output[i*O+j] + b[j];
            output[i*column+j] += bais[j];
            //printf("output[%d]=%f\n",i*column+j, output[i*column+j]);
            }
    }

}
