/**
* @fn int max(const float *input_x, int len)
* @brief 一次元配列の最大値のインデックスを返す関数
* @param[out] input_x 一次元配列
* @param[in] len 一次元配列の要素数
* @return float 最大値のインデックス
*/

#include <stdio.h>
#include "max.h"

int max(const float *input_x, int len){
  int i;
  int max_index = 0;

  for(i=1;i<len;i++){
    if(input_x[i] > input_x[max_index]){
      max_index = i;
      }
  }
  return max_index;
}
