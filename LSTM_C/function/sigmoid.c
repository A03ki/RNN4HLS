#include <stdio.h>
#include <math.h>
#include "sigmoid.h"

float sigmoid(float x){
    return (1 / (1 + exp(-x)));
}
