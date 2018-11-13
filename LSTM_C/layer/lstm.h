#ifndef LSTM_H
#define LSTM_H

/*
lstm : 一次元配列をLSTMレイヤに通す関数

Parameters
----------
output : array of pointer to float
  一次元配列(出力用)
input_x : array of pointer to float
  一次元配列(入力用)
row : int
  input_xが二次元配列のときの列の長さ

Return
------
 : void

Note
----
二次元配列を一次元配列にしたものを入力として入れる
*/

void lstm(float*, float*, int);

#endif
