# Newest CPython slim image that still has PyTorch wheels (3.15 has none yet).
FROM python:3.14.8-slim

# Pythonのバイトコード生成抑制とログ出力を即時にする設定
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# コンテナ内の作業ディレクトリを /app に設定
WORKDIR /app

# リポジトリ内の全ファイルをコンテナにコピー
COPY . .

# pipをアップグレードし、requirements_for_docker.txt のパッケージをインストール
RUN pip install --upgrade pip && pip install -r requirements_for_docker.txt

# ENTRYPOINT により、常に python src/parcellation.py が実行されるように設定
ENTRYPOINT ["python", "src/parcellation.py"]

# CMD でデフォルトの引数を設定（docker run 時に上書き可能）
CMD ["-i", "input", "-o", "output", "-m", "model"]