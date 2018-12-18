/**
* @fn void lstm(float *output, const float *input_x, int row)
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
#include "lstm.h"


/*
バイアス, 重みをdeffine.hに移す.
*/

void lstm(float *output, const float *input_x, int row){
    int matrix_k, column, t, i, j, k, l, m;
    matrix_k = 2, column = 2, m = 0;

    // 初期化
    float array_fgio[4*6] = {0};
    float array_c[3] = {0};

    // 重み定義
    int w_rowcom = 2*2;
    float weight_fgio_h[4*2*2] = {1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0};
    float weight_fgio_x[4*2*2] = {1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0};
    // バイアス定義
    int b_rowcom = 2;
    float bais_fgio[4*2] = {0.2, 0.3, 0.2, 0.3, 0.2, 0.3, 0.2, 0.3};
    // アフィン変換
    //affine(array_fgio, input_x, weight_fgio_x, bais_fgio, row, matrix_k, column);
    //affine(array_g, input_x, weight_g_x, bais_g, row, matrix_k, column);
    //affine(array_i, input_x, weight_i_x, bais_i, row, matrix_k, column);
    //affine(array_o, input_x, weight_o_x, bais_o, row, matrix_k, column);

    // アフィン変換
    int fgio_rowcom = 6;
    int in_x_rowcom = 6;
    for(i=0;i<row;i++){
      for(k=0;k<matrix_k;k++){
        for(j=0;j<column;j++){
          // f
          array_fgio[i*column+j] += input_x[i*matrix_k+k] * weight_fgio_x[k*column+j];
          // g
          array_fgio[fgio_rowcom+i*column+j] += input_x[i*matrix_k+k] * weight_fgio_x[w_rowcom+k*column+j];
          //printf("array_fgio[%d] += input_x[%d] * weight_fgio_x[%d]\n", i*column+j, in_x_rowcom+i*matrix_k+k, w_rowcom+k*column+j);
          //printf("%f += %f * %f\n",array_fgio[fgio_rowcom+i*column+j],input_x[i*matrix_k+k],weight_fgio_x[w_rowcom+k*column+j]);
          // i
          array_fgio[2*fgio_rowcom+i*column+j] += input_x[i*matrix_k+k] * weight_fgio_x[2*w_rowcom+k*column+j];
          // o
          array_fgio[3*fgio_rowcom+i*column+j] += input_x[i*matrix_k+k] * weight_fgio_x[3*w_rowcom+k*column+j];
          // 確認用
          //printf("(i, j, k)=(%d, %d, %d) ", i, j, k);
          //printf("output[%d]=%f\n",i*O+j, output[i*O+j]);
        }
      }
    }

    // バイアス加算
    //printf("Affine\n");
    for(i=0;i<row;i++){
      for(j=0;j<column;j++){
        // 確認用
        //printf("output[%d]=%f\n",i*column+j, output[i*column+j]);
        //output[i*O+j] = output[i*O+j] + b[j];
        array_fgio[i*column+j] += bais_fgio[j];
        array_fgio[fgio_rowcom+i*column+j] += bais_fgio[b_rowcom+j];
        array_fgio[2*fgio_rowcom+i*column+j] += bais_fgio[2*b_rowcom+j];
        array_fgio[3*fgio_rowcom+i*column+j] += bais_fgio[3*b_rowcom+j];
        // 確認用
        //printf("output[%d]=%f\n",i*column+j, output[i*column+j]);
      }
    }

    /*
    // 行列積の確認
    for(t=0;t<(4*row*column);t++){
        printf("array_fgio[%d]=%f\n", t, array_fgio[t]);
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
                array_fgio[i*column+j] += output[m+k] * weight_fgio_h[k*column+j];
                array_fgio[fgio_rowcom+i*column+j] += output[m+k] * weight_fgio_h[w_rowcom+k*column+j];
                array_fgio[2*fgio_rowcom+i*column+j] += output[m+k] * weight_fgio_h[2*w_rowcom+k*column+j];
                array_fgio[3*fgio_rowcom+i*column+j] += output[m+k] * weight_fgio_h[3*w_rowcom+k*column+j];
                //printf("array_f[%d]=%f\n",i*column+j, array_f[i*column+j]);
                //printf("array_g[%d]=%f\n",i*column+j, array_g[i*column+j]);
                //printf("array_i[%d]=%f\n",i*column+j, array_i[i*column+j]);
                //printf("array_o[%d]=%f\n",i*column+j, array_o[i*column+j]);
            }
        }
        for(l=0;l<column;l++){
            // printf("%d\n", l);
            //printf("array_f1[%d]=%f\n",i*column+l, array_f[i*column+l]);
            //printf("array_g[%d]=%f\n",i*column+l, array_g[i*column+l]);
            //printf("array_i[%d]=%f\n",i*column+l, array_i[i*column+l]);
            //printf("array_o[%d]=%f\n",i*column+l, array_o[i*column+l]);
            array_fgio[i*column+l] = sigmoid(array_fgio[i*column+l]);
            //printf("array_f2[%d]=%f\n",i*column+l, array_f[i*column+l]);

            array_fgio[fgio_rowcom+i*column+l] = tanhf(array_fgio[fgio_rowcom+i*column+l]);
            array_fgio[2*fgio_rowcom+i*column+l] = sigmoid(array_fgio[2*fgio_rowcom+i*column+l]);
            array_fgio[3*fgio_rowcom+i*column+l] = sigmoid(array_fgio[3*fgio_rowcom+i*column+l]);
            // printf("(%d, %d)\n", l, l);
            array_c[l] = array_fgio[i*column+l] * array_c[l] + array_fgio[fgio_rowcom+i*column+l] * array_fgio[2*fgio_rowcom+i*column+l];
            // printf("(%d, %d)\n", i*column+l, m+l);
            output[i*column+l] = array_fgio[3*fgio_rowcom+i*column+l] * tanhf(array_c[l]);

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
}
