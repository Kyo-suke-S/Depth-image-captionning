# Depth-image-captionning
## 卒業論文「画像キャプショニングにおける注意機構と奥行き情報の活用」<br>の研究で作成したコード(整理版)

## 概要

  画像キャプショニングにおける奥行き情報の効果を研究した。注意機構を導入したモデル([Show, Attdnd and Tell](https://arxiv.org/abs/1502.03044))を拡張し、
  奥行き情報を活用するモデルを構築した。注意機構は、入力画像を格子上に分割した時の各部分の特徴量(アノテーションベクトル)を利用しする。そして、キャプション生成時に各ステップでRNNの隠れ状態と各アノテーションベクトルとの関連度を計算し、
  関連の大きい部分の特徴量を単語予測に使用する。
  
  本研究では図のように奥行きマップによる奥行き情報とを画像特徴量と合わせてアテンションを計算しキャプション生成を行うモデルを構築し、MSCOCO2014データセットと画像内の物体の奥行き関係と<br>
  単語の相関を明示的表したオリジナルデータで実験し評価した。
