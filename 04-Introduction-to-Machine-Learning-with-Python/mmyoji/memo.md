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
