# Public phase-results data

`phase-results.json`はGitHub Pagesで表示する生成済みpublic artifactです。直接編集しません。

更新:

```powershell
uv run python scripts/export_phase_results.py
```

検証:

```powershell
uv run python scripts/export_phase_results.py --check
```

source of truthは`src/lsteg/reporting/phase_results.py`です。公開可能なsynthetic sampleだけを
allowlistし、実在の秘密文、key、nonce、local path、private endpointを含めません。
