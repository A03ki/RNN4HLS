#ifndef AFFINE_H
#define AFFINE_H

/*
affine : Affine変換をする関数

Parameters
----------
output : array of pointer to float
  出力用の一次元配列(行列積後の要素数と合わせる)
input_x : array of pointer to float
  重みと行列積を行う一次元配列
weight : array of pointer to float
  一次元配列からなる重み
bais : array of pointer to float
   一次元配列からなるバイアス
row : int
  input_xの列の長さ
matrix_k : int
  input_kの行の長さ
column : int
  weightの行の長さ
*/

void affine(float*, float*, float*, float*, int, int, int);

#endif
