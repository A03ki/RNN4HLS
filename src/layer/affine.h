#ifndef AFFINE_H
#define AFFINE_H

/*
affine : Affine変換をする関数

Parameters
----------
output : array of pointer to float
出力用の一次元配列(行列積後の要素数と合わせる)
input_x : const array of pointer to float
重みと行列積を行う一次元配列
weight : const array of pointer to float
一次元配列からなる重み
bais : const array of pointer to float
一次元配列からなるバイアス
row : int
input_xの行の長さ
matrix_k : int
input_kの列の長さ
column : int
weightの列の長さ
 */

void affine(float *, const float *, const float *, const float *, int, int, int);

#endif
