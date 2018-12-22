/**
* @fn void lstm(float *output, const float *input_x, int row)
* @brief 一次元配列をLSTMレイヤに通す関数
* @param[out] output 出力用の一次元配列(行列積後の要素数と合わせる)
* @param[in,out] input_x 関数内の重みと行列積を行う一次元配列
* @param[in] row input_xが二次元配列の時の列の長さ
* @return void
* @detail Pytorchの重みは転置して使う
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../function/sigmoid.h"
#include "lstm.h"


/*
今までのはLSTMでは無かったので修正,
重みは全て転置させることでPytorchとの一致を確認
*/

void lstm(float *output, const float *input_x, int row){
  int matrix_k, column,i, k, j, t, l, m;
  matrix_k =3, column =3, m = 0;
  //int row = 4;
  int fgio_t_size = 1*3*4; // tのときのifgoのsize

  // 初期化
  float array_fgio[4*3*4] = {0};
  float array_c[3] = {0};
  //float output[4*3] = {0.0};

  //重み定義
  float weight_fgio_x[3*4*3] = {0.29748732,  0.27099025, -0.11877558, -0.07069314,  0.2108646 ,
     -0.05197728,  0.5047004 , -0.34869725, -0.18501103,  0.313811  ,
      0.16129082, -0.52604556, -0.25482982, -0.5435388 ,  0.2937234 ,
      0.16013438, -0.22499397,  0.08368343,  0.17966568, -0.09677294,
      0.02764446, -0.5643892 ,  0.5476488 , -0.5489495 , -0.11192599,
      0.3462469 ,  0.08026147,  0.02848172, -0.04209387, -0.0023064 ,
     -0.21500885, -0.24903467,  0.3441745 ,  0.35791123,  0.3810857 ,
     -0.27847457};
  float weight_fgio_h[3*4*3] = {0.50697803, -0.26830125,  0.4329692 ,  0.29676658, -0.16671404,
     -0.27527004,  0.5750694 , -0.38537154, -0.3732106 ,  0.5120491 ,
     -0.01118433, -0.40970066, -0.09616864,  0.5665065 ,  0.00683677,
     -0.30646914, -0.06329745,  0.31328988,  0.46279728,  0.3515845 ,
      0.37501472, -0.32363924,  0.08432633,  0.3140812 ,  0.24708247,
     -0.24427599, -0.30415818,  0.16980141, -0.55505764, -0.14034072,
     -0.02703363,  0.17918473,  0.35051036, -0.09503531, -0.43816167,
     -0.135384};

  //バイアス定義
  float bias_fgio[4*3] = {0.6211119 , -0.30144942, -0.32368398,  0.5471303 ,  0.12431368,
     0.61093354, -0.3490818 , -0.14233065, -0.2899597 , -0.29693842,
     -0.35145867, -0.3723213};

  //float input_x[4*3] = {0.0, 1.0, 2.0, 1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0};

  // アフィン変換
  for(i=0;i<row;i++){
      for(k=0;k<matrix_k;k++){
          for(j=0;j<4*column;j++){
              array_fgio[i*4*column+j] += input_x[i*matrix_k+k] * weight_fgio_x[k*4*column+j];
          }
      }
  }
  for(i=0;i<row;i++){
      for(j=0;j<fgio_t_size;j++){
          array_fgio[i*fgio_t_size+j] += bias_fgio[j];
      }
  }

  for(t=0;t<row;t++){
      for(i=0;i<t_step;i++){
          for(k=0;k<column;k++){
              for(j=0;j<4*column;j++){
                  array_fgio[t*fgio_t_size+i*4*column+j] += output[m+i*column+k] * weight_fgio_h[k*4*column+j];
              }
          }
      }
      for(l=0;l<column;l++){
          //printf("1 array_i[%d]=%f\n",t*fgio_t_size+l, array_fgio[t*fgio_t_size+l]);
          //printf("1 array_f[%d]=%f\n",column+t*fgio_t_size+l, array_fgio[column+t*fgio_t_size+l]);
          //printf("1 array_g[%d]=%f\n",2*column+t*fgio_t_size+l, array_fgio[2*column+t*fgio_t_size+l]);
          //printf("1 array_o[%d]=%f\n",3*column+t*fgio_t_size+l, array_fgio[3*column+t*fgio_t_size+l]);
          //printf("%d\n", t*fgio_t_size+l);
          array_fgio[t*fgio_t_size+l] = sigmoid(array_fgio[t*fgio_t_size+l]);
          array_fgio[column+t*fgio_t_size+l] = sigmoid(array_fgio[column+t*fgio_t_size+l]);
          array_fgio[2*column+t*fgio_t_size+l] = tanhf(array_fgio[2*column+t*fgio_t_size+l]);
          array_fgio[3*column+t*fgio_t_size+l] = sigmoid(array_fgio[3*column+t*fgio_t_size+l]);
          array_c[l] = array_fgio[column+t*fgio_t_size+l] * array_c[l] + array_fgio[t*fgio_t_size+l] * array_fgio[2*column+t*fgio_t_size+l];
          output[t*column+l] = array_fgio[3*column+t*fgio_t_size+l] * tanhf(array_c[l]);
          //printf("2 array_i[%d]=%f\n",t*fgio_t_size+l, array_fgio[t*fgio_t_size+l]);
          //printf("2 array_f[%d]=%f\n",column+t*fgio_t_size+l, array_fgio[column+t*fgio_t_size+l]);
          //printf("2 array_g[%d]=%f\n",2*column+t*fgio_t_size+l, array_fgio[2*column+t*fgio_t_size+l]);
          //printf("2 array_o[%d]=%f\n",3*column+t*fgio_t_size+l, array_fgio[3*column+t*fgio_t_size+l]);
          //printf("array_c[%d]=%f\n",l, array_c[l]);
          //printf("output[%d]=%f\n",t*column+l, output[t*column+l]);
      }
      m = t*column;  // t-1の役割
  }
}
