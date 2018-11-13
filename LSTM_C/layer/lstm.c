/**
* @fn void lstm(float *output, float *input_x, int row)
* @brief 一次元配列をLSTMレイヤに通す関数
* @param[out] output 出力用の一次元配列(行列積後の要素数と合わせる)
* @param[in,out] input_x 関数内の重みと行列積を行う一次元配列
* @param[in] row input_xが二次元配列の時の列の長さ
* @return void
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../function/sigmoid.h"
#include "affine.h"
#include "lstm.h"

/*
ヘッダファイルの作成
M, N, Oの名前をrow, matrix_k, columnに変えた
今後やること
記憶セルをだだのキャッシュにする(余分に保持しない)
*/

void lstm(float *output, float *input_x, int row){
    int row, matrix_k, column, t, l,m,i,j,k;
    row=row, matrix_k = 2, column=2, m=0;

    float *array_f;
    float *array_g;
    float *array_i;
    float *array_o;
    float *array_c;
    array_f = malloc(sizeof(float) * row * column);
    array_g = malloc(sizeof(float) * row * column);
    array_i = malloc(sizeof(float) * row * column);
    array_o = malloc(sizeof(float) * row * column);
    array_c = malloc(sizeof(float) * row * column);

    // 初期化
    for(t=0;t<(row*column);t++){
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
    affine(array_f, input_x, weight_f_x, bais_f, row, matrix_k, column);
    affine(array_g, input_x, weight_g_x, bais_g, row, matrix_k, column);
    affine(array_i, input_x, weight_i_x, bais_i, row, matrix_k, column);
    affine(array_o, input_x, weight_o_x, bais_o, row, matrix_k, column);


    /*
    // 行列積の確認
    for(t=0;t<(row*column);t++){
        printf("array_f[%d]=%f\n", t, array_f[t]);
        printf("array_g[%d]=%f\n", t, array_g[t]);
        printf("array_i[%d]=%f\n", t, array_i[t]);
        printf("array_o[%d]=%f\n", t, array_o[t]);
    }
    */


    for(i=0;i<row;i++){
        for(k=0;k<matrix_k;k++){
            for(j=0;j<column;j++){
                //printf("(m+k, i*column+j)=(%d, %d)\n", m+k, i*column+j);
                //printf("output1[%d]=%f\n, weight_f_h[%d]=%f\n, ", m+j, output[m+j], k*column+j, weight_f_h[k*column+j]);
                //printf("array_f[%d]=%f\n",i*column+j, array_f[i*column+j]);
                //printf("weight_f_h=%f\n", weight_f_h[k*column+j]);
                //printf("output[%d]=%f\n", m+k, output[m+k]);
                array_f[i*column+j] += output[m+k] * weight_f_h[k*column+j];
                array_g[i*column+j] += output[m+k] * weight_g_h[k*column+j];
                array_i[i*column+j] += output[m+k] * weight_i_h[k*column+j];
                array_o[i*column+j] += output[m+k] * weight_o_h[k*column+j];
                //printf("array_f[%d]=%f\n",i*column+j, array_f[i*column+j]);
                //printf("array_g[%d]=%f\n",i*column+j, array_g[i*column+j]);
                //printf("array_i[%d]=%f\n",i*column+j, array_i[i*column+j]);
                //printf("array_o[%d]=%f\n",i*column+j, array_o[i*column+j]);
            }
        }
        for(l=0;l<column;l++){
            //printf("array_f1[%d]=%f\n",i*column+l, array_f[i*column+l]);
            //printf("array_g[%d]=%f\n",i*column+l, array_g[i*column+l]);
            //printf("array_i[%d]=%f\n",i*column+l, array_i[i*column+l]);
            //printf("array_o[%d]=%f\n",i*column+l, array_o[i*column+l]);
            array_f[i*column+l] = sigmoid(array_f[i*column+l]);
            //printf("array_f2[%d]=%f\n",i*column+l, array_f[i*column+l]);

            array_g[i*column+l] = tanhf(array_g[i*column+l]);
            array_i[i*column+l] = sigmoid(array_i[i*column+l]);
            array_o[i*column+l] = sigmoid(array_o[i*column+l]);
            array_c[i*column+l] = array_f[i*column+l] * array_c[m+l] + array_g[i*column+l] * array_i[i*column+l];
            output[i*column+l] = array_o[i*column+l] * tanhf(array_c[i*column+l]);

            //printf("array_f[%d]=%f\n",i*column+l, array_f[i*column+l]);
            //printf("array_g[%d]=%f\n",i*column+l, array_g[i*column+l]);
            //printf("array_i[%d]=%f\n",i*column+l, array_i[i*column+l]);
            //printf("array_o[%d]=%f\n",i*column+l, array_o[i*column+l]);
            //printf("array_c[%d]=%f\n",i*column+l, array_c[i*column+l]);
            //printf("output2[%d]=%f\n",i*column+l, output[i*column+l]);
        }
        m = i*column;  // t-1の役割
    }

    for(i=0;i<row;i++){
        for(j=0;j<column;j++){
            //printf("array_f[%d]=%f\n",i*column+j, array_f[i*column+j]);
            //printf("array_g[%d]=%f\n",i*column+j, array_g[i*column+j]);
            //printf("array_i[%d]=%f\n",i*column+j, array_i[i*column+j]);
            //printf("array_o[%d]=%f\n",i*column+j, array_o[i*column+j]);
            //printf("array_c[%d]=%f\n",i*column+j, array_c[j]);
            printf("output[%d]=%f\n",i*column+j, output[i*column+j]);
        }
    }

    // メモリ解放
    free(array_f);
    free(array_g);
    free(array_i);
    free(array_o);
}
