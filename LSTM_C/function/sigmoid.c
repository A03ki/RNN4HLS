/**
* @fn float sigmoid(float x)
* @brief 1つの入力値入れてSigmoidからの出力値を返す関数
* @param[in] input_x Sigmoidへの入力値
* @return float Singmoidからの出力値
*/

#include <stdio.h>
#include <math.h>
#include "sigmoid.h"

// sigmoid関数
float sigmoid(float x){
    return (1 / (1 + exp(-x)));
}
