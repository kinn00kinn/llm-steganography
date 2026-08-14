# 開発環境

## 1. 採用する環境

- Python 3.12.10
- uv による Python/virtual environment/dependency/lock 管理
- pytest、ruff、mypy
- `src` layout

Python 3.12 系に固定し、patch version は `.python-version` に記録する。model runtime の
対応状況を Phase 4 で再確認するまでは Python minor version を上げない。

## 2. 推奨セットアップ（Windows / macOS / Linux）

uv が入っている環境では次だけでよい。

```powershell
uv python install 3.12.10
uv sync --dev
```

`uv sync` は repository-local の `.venv` を作り、`uv.lock` と一致する dependency を
導入する。activate は必須ではなく、常に `uv run ...` を使える。

```powershell
uv run python --version
uv run steg --help
```

PowerShell で activate したい場合:

```powershell
.\.venv\Scripts\Activate.ps1
```

## 3. pyenv / pyenv-win を使う場合

この端末では pyenv は必須ではない。既に導入済みなら `.python-version` が選択に使える。

```powershell
pyenv install 3.12.10
pyenv local 3.12.10
python --version
uv sync --dev
```

Windows では通常 pyenv-win、macOS/Linux では pyenv を使う。global version は変更せず、
repository-local version のみ設定する。pyenv を新規導入して user profile や PATH を
変更する必要はなく、uv の Python 管理だけでも再現できる。

## 4. 日常コマンド

```powershell
# test
uv run pytest

# lint / import order
uv run ruff check .

# formatting verification
uv run ruff format --check .

# apply safe lint fixes and formatting
uv run ruff check --fix .
uv run ruff format .

# static types
uv run mypy src tests

# CLI
uv run steg --help
```

すべてまとめて確認する場合:

```powershell
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
```

## 5. dependency の追加

runtime dependency:

```powershell
uv add PACKAGE
```

development dependency:

```powershell
uv add --dev PACKAGE
```

手で `uv.lock` を編集しない。暗号 library は Phase 2、PyTorch/Transformers と model
artifact は Phase 4 で、採用理由と version pin を伴って追加する。GPU package は OS と
CUDA の組合せがあるため、通常の CPU-only test dependency から分離する。

## 6. test policy

- unit tests は CPU-only、offline で完結させる。
- randomized test は failure を再現できる seed/case を表示する。
- secrets、keys、平文 payload を test logs に不用意に残さない。
- LLM/GPU test は明示 marker を付け、通常の `pytest` では実行しない。
- encoder の変更には同じ条件の decoder round-trip test を必須とする。

## 7. 現在の端末で確認済みの初期状態

2026-08-14 時点で、system Python 3.12.10 と uv 0.11.32 が利用可能。pyenv は未導入。
本 repository は uv だけでセットアップできるため、global な PATH や Python installation
を変更せずに作業を開始できる。
