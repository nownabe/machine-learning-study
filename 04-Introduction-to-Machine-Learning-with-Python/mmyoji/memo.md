# Ch1

- `scipy.sparse` が重要とのこと
  - 疎行列を表現する
  - 疎行列: 成分のほとんどが 0 である行列(2次元配列)
    - ほとんどが 0 なので普通に配列データを作ると無駄が多い → CSR 方式にする
    - CSR: Compressed Sparse Row
      - http://krustf.hateblo.jp/entry/20120730/1343622560
      - CRS: Compressed Row Storage とも
      - 0 じゃない要素だけ保持する形式で、メモリに無駄がない
    - COO 方式: http://d.hatena.ne.jp/jetbead/20131107/1383751980
* `np.ones` とか `np.arange` とか API を覚えていく〜

## Problems

I executed the version-checking code in terminal, but IPython didn't work well.

```
Python version: 3.6.3 (default, Oct  3 2017, 21:45:48)
[GCC 7.2.0]

pandas version: 0.21.0

matplotlib version: 2.1.0

NumPy version: 1.13.3

SciPy version: 1.0.0

Traceback (most recent call last):
File "code.py", line 16, in <module>
import IPython
File "/usr/local/lib/python3.6/dist-packages/IPython/__init__.py", line 54, in <module>
from .core.application import Application
File "/usr/local/lib/python3.6/dist-packages/IPython/core/application.py", line 25, in <module>
from IPython.core import release, crashhandler
File "/usr/local/lib/python3.6/dist-packages/IPython/core/crashhandler.py", line 27, in <module>
from IPython.core import ultratb
File "/usr/local/lib/python3.6/dist-packages/IPython/core/ultratb.py", line 115, in <module>
from IPython.core import debugger
File "/usr/local/lib/python3.6/dist-packages/IPython/core/debugger.py", line 46, in <module>
from pdb import Pdb as OldPdb
File "/usr/lib/python3.6/pdb.py", line 76, in <module>
import code
File "/home/mmyoji/projects/mmyoji/slides/code.py", line 17, in <module>
print("IPython version: {}".format(IPython.__version__))
AttributeError: module 'IPython' has no attribute '__version__'

(IPython 6.2.1 in `python3` CLI)

scikit-learn version: 0.19.1
```

then install `jupyter` through pip.

https://jupyter.org/install.html

```bash
python3 -m pip install --upgrade pip
sudo pip3 install jupyter
```

Just run `jupyter notebook` in a directory.

## My First App

* Iris
* 機械学習では個々のアイテムを **サンプル**, その特性を **特徴量** と呼ぶ
* モデルを **作る** ためのデータと、それを **評価する** のに使うデータは別々のものでないといけない
  * 前者を **訓練データ(training data)** ないしは **訓練セット(training set)** と呼ぶ
  * 後者を **テストデータ(test data)** ないしは **テストセット(test set)** と呼ぶ
* データをよく観察する
  * おかしな値が入ってないか？単位は揃っているか
  * 散布図にしてみる
  * 特徴量の数が少なければ **ペアプロット** で組み合わせをプロットする
  * NOTE: `%matplotlib inline` と書いておけばグラフとか勝手に描画してくれる
* k-最近傍法 (k-Nearest Neighbors)
* このアプリは **教師あり学習**

# Ch2

教師あり学習について

* 入出力のペアが訓練データとなり、モデルを構築する
  * 一定量のデータが必要不可欠
* Classification クラス分類 vs Regression 回帰
* Classification
  * 1. binary classification 2クラス分類
    * Yes/No 問題 -> positive/netagtive という言い方をする
    * e.g.) メールのスパム判定
  * 2. multiclass classification 多クラス分類
    * e.g.) アイリスの花の分類, web サイトのテキストから言語を判定する
* Regression
  * floating-point number 浮動小数点数の予測
  * 出力になんらかの **連続性** があれば回帰を使う
  * e.g.)
    * 年収(の量 amount)を学歴, 年齢, 住所から推定する
    * とうもろこしの収穫量を前年の収穫量、天候、従業員数から予測
* 単純なモデルの方が、新しいデータに対してよく汎化できる
  * 逆に、過度に複雑なモデルを作ってしまうことを **過剰適合 overfitting** という
  * 単純すぎるモデルは **適合不足 underfitting**
  * 複雑さを上げると個々のデータに対しての精度は上がるが、汎用的ではなくなる
     * 精度と汎用性の tradeoff
* モデルをいじりまわすより、データを増やした方がいいケースも多い(直感的にそんな気もする)
* KNeighbors分類機
  * 近傍点の数、データポイント間の距離測度が重要な parameters
  * 実際にはほとんど使われてない
  * メリット
    * モデルの理解のしやすさ
    * あまり調整しなくても高い性能が出ることが多い
    * ベースラインとして利用可能
  * デメリット
    * 訓練セットが多くなると予測が遅くなる
    * 多数(数百以上)の特徴量を持つデータセットではうまく機能しない
    * ほとんどの特徴量が多くの場合0となるようなデータセットでは性能が悪い

