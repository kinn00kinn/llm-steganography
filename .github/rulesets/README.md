# Repository rulesets

この directory の JSON は GitHub repository rulesets の source of truth です。GitHub は
repository 内の JSON を自動適用しないため、変更は Pull Request で review した後、admin
が REST API で対応する ruleset を更新します。

現在の rules:

- `main.json`: deletion/force-push/merge commit を禁止し、PR、最新 `quality` check、
  review conversation 解決、squash merge を要求する。
- `branch-naming.json`: `main` と既定 prefix の短命 branch だけを許可する。

solo 開発中は approving review を 0 件とする。作者は自分の PR を承認できないため、1 件を
要求すると merge 不能になる。collaborator が増えたら `required_approving_review_count` を
1 へ変更する。

API へ渡す JSON は repository 名や token を含まない。適用前に GitHub 上の現行 ruleset
を取得し、同名 ruleset を重複作成しないこと。
