#ifndef LSTM_H
#define LSTM_H

/*
lstm : 一次元配列を一方向LSTMレイヤに通す関数

Parameters
----------
output : array of pointer to float
  一次元配列(出力用)
input_x : const array of pointer to float
  一次元配列(入力用)
weight_x : const array of pointer to float
  入力input_xと行列積を行う一次元配列の重み
weight_h : const array of pointer to float
  時刻t-1の時の出力outputと行列積を行う一次元配列の重み
bias : const array of pointer to float
  一次元配列のバイアス. 転置した重みの行と同じ長さを持つ
row : int
  input_xが二次元配列のときの行の長さ
matrix_k : int
  input_xが二次元配列のときの列の長さ(weight_hが二次元配列のときの行の長さ)
column : int
  weight_hが二次元配列のときの列の長さ

Return
------
 : void

Note
----
二次元配列を一次元配列にしたものを入力として入れる
Pytorchの重み(weight_ih_l0, weight_hh_l0)は転置して使う
Pytorchのバイアス(bias_ih_l0, bias_hh_l0)は足して使う
*/

void lstm(float *, const float *, const float *, const float *, const float *, int, int, int);

#endif
