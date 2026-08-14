# Security policy

## Scope

この repository は研究開発段階であり、現時点の release や generated cover text を
機密通信の保証された手段として扱わないでください。暗号 payload が完成した後も、
steganographic detectability は別途評価が必要です。

## Reporting

key disclosure、plaintext disclosure、authentication bypass、nonce reuse、decoder の
resource exhaustion、dependency compromise につながる問題を発見した場合、秘密情報や
再利用可能な key を public issue に載せないでください。

GitHub の Private vulnerability reporting が repository で有効な場合はそれを使用する。
未設定の場合は、再現用の無害な synthetic data だけで public issue を作り、機密な詳細を
送る手段の案内を待ってください。

## Handling test data

- 実在する password、token、個人情報を secret text の sample に使わない。
- key file、`.env`、raw experiment artifact を commit しない。
- GitHub Pages は入力欄や runtime API 接続を持たず、公開済み synthetic sample だけを扱う。
- sample publish 前に key、private input、内部 path、不要な metadata の redaction を検証する。
