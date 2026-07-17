![Cover](../media/Cover.png)

![Explain](../media/Explain.png)

![Explain](../media/Application.png)

# OpenMAP-T1
![Figure3](../media/Representative.png)

**言語**: [English](../README.md) | [日本語 (Japanese)](README-ja.md) | [中文 (Chinese)](README-zh.md)

[![](http://img.shields.io/badge/medRxiv-10.1101/2024.01.18.24301494-B31B1B.svg)](https://www.medrxiv.org/content/10.1101/2024.01.18.24301494v1)
[![Human Brain Mapping](https://img.shields.io/badge/Published-Human%20Brain%20Mapping-%2300629B)](https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.70063)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1fmfkxxZjChExnl5cHITYkNYgTu3MZ7Ql#scrollTo=xwZxyL5ewVNF)
![Python 3.11+](https://img.shields.io/badge/Python-3.11+-brightgreen.svg)
[![HuggingFace Application](https://img.shields.io/badge/HuggingFace-Application-orange.svg)](https://huggingface.co/spaces/OishiLab/OpenMAP-T1)

[![GitHub — 40 stars — 感謝](https://img.shields.io/badge/GitHub-40%E2%98%85%20%E6%84%9F%E8%AC%9D!-D5312A?style=for-the-badge&logo=github&logoColor=white)](https://github.com/OishiLab/OpenMAP-T1)

**OpenMAP-T1: 全脳をカバーする280の解剖学的領域を分割するための高速深層学習アプローチ**<br>
**著者**: [Kei Nishimaki](https://scholar.google.com/citations?user=iJPjUEkAAAAJ&hl=ja), [Kengo Onda](https://scholar.google.com/citations?user=GEXxaywAAAAJ&hl=en), [Kumpei Ikuta](https://scholar.google.com/citations?user=Q9h-OP8AAAAJ&hl=ja), [Jill Chotiyanonta](https://www.researchgate.net/profile/Jill-Chotiyanonta), [Yuto Uchida](https://scholar.google.com/citations?user=rKPaAF4AAAAJ&hl=ja), [Susumu Mori](https://scholar.google.com/citations?user=eiGSZdMAAAAJ&hl=en), [Hitoshi Iyatomi](https://scholar.google.com/citations?user=ghyQxvIAAAAJ&hl=ja), [Kenichi Oishi](https://www.hopkinsmedicine.org/profiles/details/kenichi-oishi)<br>

The Russell H. Morgan Department of Radiology and Radiological Science, The Johns Hopkins University School of Medicine, Baltimore, MD, USA <br>
Department of Applied Informatics, Graduate School of Science and Engineering, Hosei University, Tokyo, Japan <br>
The Richman Family Precision Medicine Center of Excellence in Alzheimer's Disease, Johns Hopkins University School of Medicine, Baltimore, MD, USA<br>

**概要**: *本研究では、従来の正規化によるアトラスベースのアプローチやマルチアトラスラベル融合（MALF）技術の限界を克服することを目的とした、T1強調脳MRIにおける迅速かつ正確な全脳パーセレーションのための深層学習ベースの手法であるOpenMAP-T1を紹介します。脳画像のパーセレーションは、特定の脳領域の詳細な分析を可能にするため、神経科学および臨床研究における基本的なプロセスです。このタスクには正規化によるアトラスベースの方法が採用されてきましたが、特に病的な状態における脳の形態の変化により限界に直面しています。MALF技術は画像のパーセレーション精度と脳形態の変化に対するロバスト性を向上させましたが、長い処理時間を要するという高い計算コストが犠牲となりました。OpenMAP-T1は、前処理、クロッピング、頭蓋除去（スカルストリッピング）、パーセレーション、半球セグメンテーション、そして最終的な統合という6つのフェーズにわたって複数の畳み込みニューラルネットワークモデルを統合しています。このプロセスには、MRI画像の標準化、脳組織の分離、そして詳細な灰白質および白質構造を含む全脳をカバーする280の解剖学的構造への分割が含まれます。同時に、パーセレーションプロセスを簡素化し、様々なスキャンタイプや条件に対応するための堅牢なトレーニングを取り入れています。OpenMAP-T1は、実世界の臨床画像を含む8つの利用可能なオープンリソースでテストされ、スキャナーの種類、磁場強度、および顔の匿名化（defacing）などの画像処理技術の違いがある様々なデータセットに対して堅牢性を示しました。既存の方法と比較して、OpenMAP-T1は精度を損なうことなく、画像あたりの処理時間を数時間から90秒未満へと大幅に短縮しました。特に、臨床現場で一般的に見られる強度の不均一性や頭部位置の変化がある画像の処理において効果的でした。OpenMAP-T1の幅広いMRIデータセットへの適応性と様々なスキャン条件への堅牢性は、ニューロイメージングにおける汎用的なツールとしての可能性を強調しています。*

論文: https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.70063<br>
クラウドアプリケーション: https://huggingface.co/spaces/OishiLab/OpenMAP-T1<br>
**Human Brain Mapping** に掲載済み<br>

## 動作要件
- **Python 3.11 以降**（pandas 3.0 の要件）
- **PyTorch 2.12 以降**（`uv sync` で自動インストール。pip 利用時は別途インストール）
- その他の固定バージョン: nibabel 5.4.2, pandas 3.0.3, scipy 1.17.1, SimpleITK 2.5.5, tqdm 4.68.2

# インストール
**OpenMAP-T1は、JHUアトラスに基づいて全脳を280の解剖学的領域に50秒/症例で分割します。**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1fmfkxxZjChExnl5cHITYkNYgTu3MZ7Ql#scrollTo=xwZxyL5ewVNF)

## uvを使用したインストール（推奨）

[uv](https://github.com/astral-sh/uv)は、Rustで書かれた非常に高速なPythonパッケージインストーラーおよびリゾルバーです。依存関係の管理において、pipの高速な代替手段を提供します。

0. uvのインストール<br>
   以下のいずれかの方法でuvをインストールします：

   **macOSとLinux:**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   **Windows:**
   ```powershell
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

   またはpipを使用：
   ```bash
   pip install uv
   ```

1. このリポジトリをクローンし、リポジトリ内に移動します：
```
git clone https://github.com/OishiLab/OpenMAP-T1.git
cd OpenMAP-T1
```

2. uvで依存関係をインストールします（PyTorch を含む）：
```
uv sync
```

3. 以下のリンクから **OpenMAP-T1 v3** の事前学習済みモデルを申請・ダウンロードし、サーバー上の `MODEL_FOLDER` に配置してください。

4. OpenMAP-T1を実行できます！！

## pipを使用したインストール（代替方法）

0. Python 3.11 以降のインストールと仮想環境の作成<br>

1. このリポジトリをクローンし、リポジトリ内に移動します：
```
git clone https://github.com/OishiLab/OpenMAP-T1.git
cd OpenMAP-T1
```
2. 環境に合わせてPyTorchをインストールしてください。<br>
https://pytorch.org/

最新の安定版 PyTorch（2.12 以降）は **Python 3.10 以降**が必要です。本プロジェクトは pandas 3.0 のため **Python 3.11 以降**が必要です。

環境を選択すると、必要なコマンドが表示されます。

![image](../media/PyTorch.png)

古いPyTorch環境をインストールしたい場合は、以下のリンクからダウンロードできます。<br>
https://pytorch.org/get-started/previous-versions/

3. PyTorch以外のライブラリをインストールします：
```
pip install -r requirements.txt
```
4. 以下のリンクから **OpenMAP-T1 v3** の事前学習済みモデルを申請・ダウンロードし、サーバー上の `MODEL_FOLDER` に配置してください。

5. OpenMAP-T1を実行できます！！

# Docker インストール手順
Docker イメージは `python:3.13.2-slim` をベースとし、`uv.lock` からエクスポートした `requirements_for_docker.txt` を使用して依存関係をインストールします。依存関係を更新した場合は、次のコマンドで再生成できます：
```
uv export --no-hashes --no-emit-project --no-dev -o requirements_for_docker.txt
```

1. Dockerイメージのビルド

要約すると、このコマンドは `Dockerfile` と現在のディレクトリ内のファイルに基づいて "openmap-t1" という名前のDockerイメージを作成します。
```
docker build -t openmap-t1 .
```
* ```docker build```: このコマンドは `Dockerfile` に記載された指示に従ってDockerイメージをビルドします。
* ```-t openmap-t1```: -t フラグはイメージに "openmap-t1" という名前（タグ）を付けます。これにより後で参照しやすくなります。
* ```.```: ドットはビルドコンテキストを表し、Dockerが `Dockerfile` やその他の必要なファイルを現在のディレクトリで探すことを意味します。

2. Dockerコンテナの実行
```
docker run --rm -it -v "$(pwd):/app" openmap-t1 -i INPUT_FOLDER -o OUTPUT_FOLDER -m MODEL_FOLDER
```
* ```docker run```: Dockerイメージから新しいコンテナを開始します。
* ```--rm```: 実行が停止したときに自動的にコンテナを削除し、停止したコンテナを残さないようにしてシステムをクリーンに保ちます。

* ```-it```
2つのオプションを組み合わせています：
   * ```-i``` STDINを開いたままにします（インタラクティブモード）。
   * ```-t``` 疑似TTYを割り当てます（端末のような対話を可能にします）。

これらを合わせることで、必要に応じて端末を通じてコンテナと対話できるようになります。

* ```-v "$(pwd):/app"```: 現在の作業ディレクトリ（$(pwd)の結果）をコンテナ内の `/app` ディレクトリにマウントします。これは以下のことを意味します：
   * 現在のフォルダ内のすべてのファイルがコンテナ内の `/app` で利用可能になります。
   * コンテナ内で行われた変更（出力ファイルなど）はホストシステムに反映されます。

* ```openmap-t1```: コンテナの作成元となるDockerイメージの名前です。これは以前に ```docker build -t openmap-t1 .``` コマンドを使用してビルドされている必要があります。

* ```-i INPUT_FOLDER -o OUTPUT_FOLDER -m MODEL_FOLDER```
これらはコンテナ内で実行されるアプリケーションに渡されるコマンドライン引数です：
   * ```-i INPUT_FOLDER```: 入力フォルダのパスを指定します。
   * ```-o OUTPUT_FOLDER```: 出力フォルダのパスを指定します。
   * ```-m MODEL_FOLDER```: モデルフォルダのパスを指定します。
   ```INPUT_FOLDER```、```OUTPUT_FOLDER```、および ```MODEL_FOLDER``` を、マウントされた ```/app``` ディレクトリ内に存在する適切なディレクトリ名またはパスに置き換えてください（例: `input`、`output`、`model`）。

# 事前学習済みモデルのダウンロード方法
事前学習済みモデルはこのリンクから入手できます。
[事前学習済みモデルのリンク](https://forms.office.com/Pages/ResponsePage.aspx?id=OPSkn-axO0eAP4b4rt8N7Iz6VabmlEBIhG4j3FiMk75UQUxBMkVPTzlIQTQ1UEZJSFY1NURDNzRERC4u)

![image](../media/Download_pretrained.png)

# すべてのコマンド
OpenMAP-T1の使用は簡単です。Linux、macOS、または Windows のターミナルから使用できます。CPUおよびGPUのサポートを提供しています。GPUでの実行ははるかに高速であり、常に推奨されます。以下はOpenMAP-T1の使用方法の最小限の例です。

## 基本的な使用法
ターミナルから以下のコマンドを実行します：
```
# uv（uv syncでインストールした場合）
uv run python src/parcellation.py -i INPUT_FOLDER -o OUTPUT_FOLDER -m MODEL_FOLDER
```
```
# pip（事前に仮想環境を有効化: source .venv/bin/activate）
python3 src/parcellation.py -i INPUT_FOLDER -o OUTPUT_FOLDER -m MODEL_FOLDER
```
```
# Docker
docker run --rm -it -v "$(pwd):/app" openmap-t1 -i INPUT_FOLDER -o OUTPUT_FOLDER -m MODEL_FOLDER
```
```
# 出力を .nii で保存する場合（任意。デフォルトは .nii.gz）
python3 src/parcellation.py -i INPUT_FOLDER -o OUTPUT_FOLDER -m MODEL_FOLDER --output-ext .nii
```
* **-i INPUT_FOLDER**: 入力脳MRI画像が含まれるフォルダを指定します。
* **-o OUTPUT_FOLDER**: 結果が保存されるフォルダを定義します。このフォルダが存在しない場合は自動的に作成されます。
* **-m MODEL_FOLDER**: 処理用の事前学習済みモデルが含まれるフォルダを指定します。
* **--output-ext {.nii.gz, .nii}**: 生成されるNIfTI画像の拡張子を指定します。デフォルトは `.nii.gz` です。
* **--output-space {native, conform, both}**: `cropped/`・`stripped/`・`parcellated/` に保存するNIfTIのボクセルグリッドを指定します。デフォルトは `native` です。
   * `native`: N4補正後の canonical 入力グリッドにリサンプルして保存（従来どおりのファイル名）。
   * `conform`: 1 mm 等方・256³ の処理グリッドのまま保存。ファイル名に `_1mm` サフィックスを付与。
   * `both`: 両方を保存。1 mm 版は `conform/` 配下に `_1mm` 付きで出力。

## 出力空間（Output Space）
推論は内部で 1 mm 等方グリッド上で行われます。デフォルトでは、パーセレーション画像と中間マスクは N4 補正後の canonical 入力ジオメトリ（`native`）に戻して保存されます。1 mm グリッドのまま保存する、または両方保存する場合は `--output-space` を指定します：
```
# uv
uv run python src/parcellation.py -i INPUT_FOLDER -o OUTPUT_FOLDER -m MODEL_FOLDER --output-space conform
uv run python src/parcellation.py -i INPUT_FOLDER -o OUTPUT_FOLDER -m MODEL_FOLDER --output-space both
```
```
# pip
python3 src/parcellation.py -i INPUT_FOLDER -o OUTPUT_FOLDER -m MODEL_FOLDER --output-space conform
python3 src/parcellation.py -i INPUT_FOLDER -o OUTPUT_FOLDER -m MODEL_FOLDER --output-space both
```
1 mm 出力のファイル名には `_1mm` が付きます（例: `A_Type1_Level5_1mm.nii.gz`、`A_cropped_mask_1mm.nii.gz`）。`original/` フォルダは常に入力由来のジオメトリで保存されます。CSV の領域体積は、このオプションに関係なく常に 1 mm グリッド上で計算されます。

## オプションの高速処理ステップ
OpenMAP-T1では、以下の排他的なフラグを使用して特定の処理ステップのみを実行できるようになりました。これらのオプションを指定することで、OpenMAP-T1は不要な処理ステップをスキップし、全体の処理時間を大幅に短縮できます。

* **顔のクロッピングのみ (Only Face Cropping)**: 顔のクロッピングのみを行い、残りの処理ステップをスキップしたい場合に使用します：
```
# uv
uv run python src/parcellation.py -i INPUT_FOLDER -o OUTPUT_FOLDER -m MODEL_FOLDER --only-face-cropping
```
```
# pip
python3 src/parcellation.py -i INPUT_FOLDER -o OUTPUT_FOLDER -m MODEL_FOLDER --only-face-cropping
```
```
# Docker
docker run --rm -it -v "$(pwd):/app" openmap-t1 -i INPUT_FOLDER -o OUTPUT_FOLDER -m MODEL_FOLDER --only-face-cropping
```
* **頭蓋除去のみ (Only Skull Stripping)**: 頭蓋除去（スカルストリッピング）のみを行い、他のすべての処理ステップをスキップしたい場合は、スカルストリッピングフラグを使用します。スカルストリッピングには前提条件として顔のクロッピングが必要であるため、顔のクロッピングはスキップされる「他の処理」ステップには含まれないことに注意してください：
```
# uv
uv run python src/parcellation.py -i INPUT_FOLDER -o OUTPUT_FOLDER -m MODEL_FOLDER --only-skull-stripping
```
```
# pip
python3 src/parcellation.py -i INPUT_FOLDER -o OUTPUT_FOLDER -m MODEL_FOLDER --only-skull-stripping
```
```
# Docker
docker run --rm -it -v "$(pwd):/app" openmap-t1 -i INPUT_FOLDER -o OUTPUT_FOLDER -m MODEL_FOLDER --only-skull-stripping
```

## GPU の使用
OpenMAP-T1 は GPU が利用可能な場合、起動時に自動的に CUDA を選択します（`Using device: cuda` と表示）。可能な限り GPU での実行を推奨します。1 回の実行で使う GPU は 1 枚です。

**Linux または Windows（WSL2）+ NVIDIA GPU** で Docker 内の GPU を使うには、事前に [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) のインストールが必要です。`--gpus` を付けないと、GPU マシン上でもコンテナ内は CPU 落ちになることがあります。

Docker で GPU を使う例:
```
docker run --rm -it --gpus all -v "$(pwd):/app" openmap-t1 -i INPUT_FOLDER -o OUTPUT_FOLDER -m MODEL_FOLDER
```

**特定の GPU** をローカル実行で使う場合（例: GPU 1）は、コマンドの前に ```CUDA_VISIBLE_DEVICES=N``` を付けます:
```
# uv
CUDA_VISIBLE_DEVICES=1 uv run python src/parcellation.py -i INPUT_FOLDER -o OUTPUT_FOLDER -m MODEL_FOLDER
```
```
# pip
CUDA_VISIBLE_DEVICES=1 python3 src/parcellation.py -i INPUT_FOLDER -o OUTPUT_FOLDER -m MODEL_FOLDER
```

Docker でも同様に指定できます（例: ホストの GPU 1）:
```
docker run --rm -it --gpus '"device=1"' -v "$(pwd):/app" openmap-t1 -i INPUT_FOLDER -o OUTPUT_FOLDER -m MODEL_FOLDER
```

**macOS** では Docker は Linux VM 上で動作するため、ホストの Apple Silicon GPU（MPS）は利用できません。Mac で MPS を使う場合はネイティブインストール（`uv` または `pip`）を、Docker は CPU 実行を想定してください。

Windowsユーザーでエラーが発生する場合は、 ```Python3``` を ```Python``` に変更してください。

# フォルダ構成
入力画像はNIfTI形式（`.nii` または `.nii.gz`）である必要があります。生成されるNIfTI出力はデフォルトで `.nii.gz` です（`.nii` で保存したい場合は `--output-ext .nii` を指定）。

以下のツリーは `--output-space native`（N4 補正後の canonical 入力グリッドに戻して保存）のデフォルト構成です。`--output-space conform` では同じサブフォルダ名を使いますが、1 mm グリッドのまま保存され、ファイル名に `_1mm` が付きます（例: `A_Type1_Level5_1mm.nii.gz`）。`--output-space both` では、下記の native 構成に加えて、1 mm 版が `conform/` 配下に保存されます（例: `conform/parcellated/A_Type1_Level5_1mm.nii.gz`）。
```
INPUT_FOLDER/
   ├ A.nii or .nii.gz
   ├ B.nii or .nii.gz
   ├ *.nii or .nii.gz

OUTPUT_FOLDER/
   ├── failed_cases.csv
   ├── A
   │   ├── cropped
   │   │   ├── A_cropped_mask.nii.gz
   │   │   └── A_cropped.nii.gz
   │   ├── csv
   │   │   ├── A_Type1_Level1.csv
   │   │   ├── A_Type1_Level2.csv
   │   │   ├── A_Type1_Level3.csv
   │   │   ├── A_Type1_Level4.csv
   │   │   ├── A_Type1_Level5.csv
   │   │   ├── A_Type2_Level1.csv
   │   │   ├── A_Type2_Level2.csv
   │   │   ├── A_Type2_Level3.csv
   │   │   ├── A_Type2_Level4.csv
   │   │   └── A_Type2_Level5.csv
   │   ├── original
   │   │   ├── A_N4.nii.gz
   │   │   └── A.nii.gz
   │   ├── parcellated
   │   │   ├── A_Type1_Level1.nii.gz
   │   │   ├── A_Type1_Level2.nii.gz
   │   │   ├── A_Type1_Level3.nii.gz
   │   │   ├── A_Type1_Level4.nii.gz
   │   │   ├── A_Type1_Level5.nii.gz
   │   │   ├── A_Type2_Level1.nii.gz
   │   │   ├── A_Type2_Level2.nii.gz
   │   │   ├── A_Type2_Level3.nii.gz
   │   │   ├── A_Type2_Level4.nii.gz
   │   │   └── A_Type2_Level5.nii.gz
   │   ├── stripped
   │   │   ├── A_stripped_mask.nii.gz
   │   │   └── A_stripped.nii.gz
   │   └── conform/                 # --output-space both のときのみ
   │       ├── cropped/
   │       │   ├── A_cropped_mask_1mm.nii.gz
   │       │   └── A_cropped_1mm.nii.gz
   │       ├── parcellated/
   │       │   └── A_Type1_Level5_1mm.nii.gz
   │       └── stripped/
   │           ├── A_stripped_mask_1mm.nii.gz
   │           └── A_stripped_1mm.nii.gz
   ├── ...

MODEL_FOLDER/
   ├ CNet/CNet.pth
   ├ SSNet/SSNet.pth
   ├ PNet/PNet.pth
   └ HNet/HNet.pth
```

配布されている事前学習済みモデルは、各ネットワークに1つの重みファイルを使用します。PNet と HNet は、推論時に断面方向をエンコードする単一モデルで全断面を処理します。

処理に失敗した症例、短絡フラグでスキップした症例、または全脳体積が明らかに異常に小さい症例（デフォルト: 1 mm 等方グリッド上で 10,000 mm³ 未満）は、`OUTPUT_FOLDER` 直下の `failed_cases.csv` に記録されます。列は `case_id`, `input_path`, `status`（`failed`, `skipped`, `low_volume`）, `reason`, `total_brain_volume_mm3` です。

## Levelメタデータ（JHU-Atlas）
[`level/`](../level/) フォルダには、OpenMAP-T1 の JHU-atlas ベースのパーセレーションに対応する階層定義・対応表が含まれています。

- `level/OpenMAP-T1_multilevel_lookup_table_dictionary.csv` には、**Type1 Level5** の正式ROI名称と、各階層レベルへの対応情報が記載されています。
- Type1 のテキスト辞書は `level/Type1Level1.txt` から `level/Type1Level5.txt` という命名です。

# 補足情報
![supplementary_level](../media/Multilevel.png)
OpenMAP-T1は、脳全体を5つの階層的な構造レベルに分割します。最も粗いレベルは8つの構造からなり、最も細かいレベルは280の構造からなります。

* 追加の可視化と詳細な分析のために、[3D Slicer](https://www.slicer.org/) も利用できます。3D Slicerは、セグメンテーション、レジストレーション、3D可視化のための堅牢なツールを提供する、医療画像コンピューティングのための無料のオープンソースプラットフォームであり、OpenMAP-T1によって生成されたパーセレーションマップを探索するのに最適な選択肢です。

* 追加の可視化と詳細な分析のために、[ROIEditor](https://www.mristudio.org/installation.html) も優れたツールです。ROIEditorは、医療画像における関心領域（ROI）の作成と編集のために特別に設計された無料のオープンソースアプリケーションです。そのユーザーフレンドリーなインターフェースは正確なセグメンテーションと微調整を容易にし、OpenMAP-T1によって生成されたパーセレーションマップ上の特定の領域を分離して分析するのに理想的です。

# FAQ
* **OpenMAP-T1の貢献は何ですか？**<br>
GPUを使用する場合、OpenMAP-T1はT1強調画像をわずか1〜2分で処理します。CPUでも処理はおよそ30分で完了します。さらに、OpenMAP-T1は撮影環境の違いや脳萎縮に対して非常に高い堅牢性を持っています。

* **OpenMAP-T1ではどのアトラスが使用されていますか？**<br>
OpenMAP-T1はJHUアトラスを採用しています。

* **OpenMAP-T1は2D画像データで使用できますか？**<br>
論文には含まれていませんが、私たちの内部テストではOpenMAP-T1が2D画像でも機能することを示しています。ただし、2Dデータを使用すると一部の領域が失われる可能性があります。

* **OpenMAP-T1を実行するにはどれくらいのGPUメモリが必要ですか？**<br>
私たちはすべての実験を24 GBメモリを搭載したNVIDIA RTX3090 GPUで行いました。推論にはそれほど必要ありませんが、推論はCNNの完全畳み込みの性質を利用して実装されているため、必要なメモリ量は画像に依存します。典型的な画像であれば、4 GB未満のGPUメモリ消費で実行できるはずです。メモリ不足の問題が発生した場合は、以下を確認してください：1）データのボクセル間隔が正しいこと、2）MRI画像に頭部領域のみが含まれていること。

* **MacBookのApple Silicon（MPS）で利用できますか？**<br>
はい。OpenMAP-T1はApple Silicon Mac（M1/M2/M3/M4）搭載のMacBookでMPS（Metal Performance Shaders）をサポートしています。Apple Silicon搭載のMacBookで実行すると、利用可能な場合は自動的にMPSによるGPUアクセラレーションが使用され、CPUのみの実行より高速に処理されます。

* **Type 1とType 2の違いは何ですか？**<br>
Type 1は脳の表面構造を皮質と皮質下白質に分離して出力します。通常、ほとんどの場合でType 1が使用されます。対照的に、Type 2は皮質と皮質下白質を単一のROIとして出力します。Type 2は、データ分析においてROIの数を最小限に抑えたい場合や、低解像度のPET画像を使用して局所脳血流量などのパラメータを定量化する場合に採用されることがあります。

* **CSVに出力される各領域の体積の単位は何ですか。また、なぜ小数点以下がありませんか。**<br>
単位は立方ミリメートル（mm³）です。値が小数のない整数になるのは、前処理で画像を 1×1×1 mm の等方向ボクセルに合わせた（conform した）座標系で、各ROIのボクセル数を数えて体積を求めているためです。このタイミングでは各ボクセルの物理体積が 1 mm³ に対応するため、体積は整数で表されます。デフォルトでは NIfTI パーセレーション画像は N4 補正後の canonical 入力グリッドに戻して保存されます（`--output-space native`）。CSV と同じ 1 mm グリッドの NIfTI も欲しい場合は `--output-space conform` または `both` を指定してください（1 mm 版のファイル名には `_1mm` が付きます）。

* **`failed_cases.csv` には何が記録されますか。**<br>
バッチ処理の終了時に、失敗した症例、`--only-face-cropping` / `--only-skull-stripping` でスキップした症例、または全脳ラベル体積が異常に小さい症例（デフォルト 10,000 mm³ 未満）が1件以上ある場合、`OUTPUT_FOLDER` 直下に `failed_cases.csv` が出力されます。

* **トレーニングコードも提供されますか？**<br>
いいえ。トレーニングコードは公開できないデータと密接に関連しているためです。

# 引用
```
@article{nishimaki2024openmap,
  author={Nishimaki, Kei and Onda, Kengo and Ikuta, Kumpei and Chotiyanonta, Jill and Uchida, Yuto and Mori, Susumu and Iyatomi, Hitoshi and Oishi, Kenichi and Alzheimer's Disease Neuroimaging Initiative and {Australian Imaging Biomarkers and Lifestyle Flagship Study of Ageing}},
  title={OpenMAP-T1: A Rapid Deep-Learning Approach to Parcellate 280 Anatomical Regions to Cover the Whole Brain},
  journal={Human Brain Mapping},
  volume={45},
  number={16},
  pages={e70063},
  keywords={brain, deep-learning, MRI, parcellation, segmentation, T1},
  doi={10.1002/hbm.70063},
  url={https://onlinelibrary.wiley.com/doi/abs/10.1002/hbm.70063},
  eprint={https://onlinelibrary.wiley.com/doi/pdf/10.1002/hbm.70063},
  note={e70063 HBM-24-0380.R2},
  abstract={ABSTRACT This study introduces OpenMAP-T1, a deep-learning-based method for rapid and accurate whole-brain parcellation in T1-weighted brain MRI, which aims to overcome the limitations of conventional normalization-to-atlas-based approaches and multi-atlas label-fusion (MALF) techniques. Brain image parcellation is a fundamental process in neuroscientific and clinical research, enabling a detailed analysis of specific cerebral regions. Normalization-to-atlas-based methods have been employed for this task, but they face limitations due to variations in brain morphology, especially in pathological conditions. The MALF techniques improved the accuracy of the image parcellation and robustness to variations in brain morphology, but at the cost of high computational demand that requires a lengthy processing time. OpenMAP-T1 integrates several convolutional neural network models across six phases: preprocessing; cropping; skull-stripping; parcellation; hemisphere segmentation; and final merging. This process involves standardizing MRI images, isolating the brain tissue, and parcellating it into 280 anatomical structures that cover the whole brain, including detailed gray and white matter structures, while simplifying the parcellation processes and incorporating robust training to handle various scan types and conditions. The OpenMAP-T1 was validated on the Johns Hopkins University atlas library and eight available open resources, including real-world clinical images, and the demonstration of robustness across different datasets with variations in scanner types, magnetic field strengths, and image processing techniques, such as defacing. Compared with existing methods, OpenMAP-T1 significantly reduced the processing time per image from several hours to less than 90 s without compromising accuracy. It was particularly effective in handling images with intensity inhomogeneity and varying head positions, conditions commonly seen in clinical settings. The adaptability of OpenMAP-T1 to a wide range of MRI datasets and its robustness to various scan conditions highlight its potential as a versatile tool in neuroimaging.},
  year={2024}
}
```

# 関連研究
以下の研究では、T1強調MRIにおける高度なセグメンテーションと分析にOpenMAP-T1が利用されています。
* **[Data-driven schizophrenia subtyping via brain atrophy trajectories and functional connectivity](https://www.nature.com/articles/s41398-026-03968-w)**  
   *著者:* Daisuke Yoshimaru, Kazuya Ouchi, Shuhei Shibukawa, Masakazu Ozawa, Motoki Hirabayashi, Ami Yuzawa, Tomokazu Tsurugizawa, Kenichi Oishi, Hirotaka James Okano;  
   *ジャーナル:* Translational Psychiatry
* **[Change points for dynamic biomarkers in the Alzheimer's disease pathological cascade: A 30-year cohort study](https://alz-journals.onlinelibrary.wiley.com/doi/full/10.1002/alz.70544)**  
   *著者:* Yuto Uchida, Kei Nishimaki, Anja Soldan, Corinne Pettigrew, Sara Grace Ho, Abhay Moghekar, Mei-Cheng Wang, Michael I. Miller, Marilyn Albert, Kenichi Oishi;
   *ジャーナル:* Alzheimer's & dementia
* **[Acceleration of Brain Atrophy and Progression From Normal Cognition to Mild Cognitive Impairment](https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2825474)**  
   *著者:* Yuto Uchida, MD, PhD; Kei Nishimaki; Anja Soldan, PhD; Abhay Moghekar, MBBS; Marilyn Albert, PhD; Kenichi Oishi, MD, PhD;  
   *ジャーナル:* JAMA Network Open
* **[A Neural Network Approach to Identify Left–Right Orientation of Anatomical Brain MRI](https://pmc.ncbi.nlm.nih.gov/articles/PMC11808181/)**  
   *著者:* Kei Nishimaki; Hitoshi Iyatomi, PhD; Kenichi Oishi, MD, PhD;  
   *ジャーナル:* Brain and Behavior
