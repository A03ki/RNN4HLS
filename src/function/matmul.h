#ifndef MATMUL_H
#define MATMUL_H

/*
matmul : 行列積

Parameters
----------
output : array of pointer to float
  出力用の一次元配列(行列積後の要素数と合わせる)
input_x : const array of pointer to float
  重みと行列積を行う一次元配列
weight : const array of pointer to float
  一次元配列からなる重み
row : int
  input_xの列の長さ
matrix_k : int
  input_kの行の長さ
column : int
  weightの行の長さ
*/

void matmul(float *, const float *, const float *, int, int, int);

#endif
