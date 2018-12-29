#include <stdio.h>
#include <stdlib.h>
//#include "layer/affine.h"
#include "layer/lstm.h"
//#include "layer/max.h"

int main(void){
  int row, column, t;
  row = 4; column = 3;
  float input_x[4*3] = {0.0, 1.0, 2.0, 1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0};
  // 初期化
  float output_x[4*3] = {0.0};

  lstm(output_x, input_x, row); // 重みは3×3

  // 確認用
  for(t=0;t<row*column;t++){
    printf("%f\n", output_x[t]);
  }
  return 0;
}
