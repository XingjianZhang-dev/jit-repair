#!/usr/bin/env bash
# Downloads, checks and unpacks the benchmarks and the official evaluators into $JIT_WORK/data (see DATA.md).
# Dr.Spider and ProofWriter are obtained from their authors' repositories as described in DATA.md.
set -euo pipefail
WORK="${JIT_WORK:-/workspace}"
D="$WORK/data"; mkdir -p "$D/dl" "$D/spider" "$D/bird" "$D/testsuite"
cd "$D/dl"
python -m pip install -q gdown
check() { obs=$(sha256sum "$2" | awk '{print $1}'); [ "$obs" = "$1" ] && echo "HASH_OK $2" || { echo "HASH_MISMATCH $2 $obs"; exit 1; }; }
[ -f dev.zip ] || curl -sSL -o dev.zip https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip
check cdd6d19faeb45a23970b98d3ef6c40a87987c95459c2cf12076897a60cf5a630 dev.zip
[ -f spider_data.zip ] || gdown --fuzzy "https://drive.google.com/uc?id=1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J" -O spider_data.zip
check 00636695dabed6b5f4b8328a16b13e069a2f16591d5efcce57660669c85b121b spider_data.zip
[ -d test-suite-sql-eval ] || git clone -q https://github.com/taoyds/test-suite-sql-eval.git
git -C test-suite-sql-eval checkout -q e97acc546ecbee8fa27fa8dbf025ef61493a876c
check 7401e4014a8955376a7919c06903a7f0ab403c99e89f94204cd8f4c8e32ae779 test-suite-sql-eval/evaluation.py
check 927fc564f7a8e34f09f009a2f5564a83fdf95226440dde84c87871fd65fe55a1 test-suite-sql-eval/process_sql.py
[ -d mini_dev ] || git clone -q https://github.com/bird-bench/mini_dev.git
git -C mini_dev checkout -q b3d4bcbbae9a96934ad812551eb400c7a3b23c12
[ -f testsuitedatabases.zip ] || gdown --fuzzy "https://drive.google.com/uc?id=1mkCx2GOFIqNesD4y8TDAO1yX1QZORP5w" -O testsuitedatabases.zip
check 9ec24ea8debc6bd04abfe137b5f1a739b5a8836f32c0464e4dfc94eb7f41da96 testsuitedatabases.zip
(cd "$D/spider" && unzip -q -o "$D/dl/spider_data.zip")
(cd "$D/bird" && unzip -q -o "$D/dl/dev.zip")
for z in $(find "$D/bird" -name "dev_databases.zip"); do (cd "$(dirname "$z")" && unzip -q -o "$z"); done
(cd "$D/testsuite" && unzip -q -o "$D/dl/testsuitedatabases.zip")
echo DATA_READY
