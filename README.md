Broad Match
=============================

### Сборка jar файла
1. maven install commons
2. maven install ml
3. maven package experiments

[![Build Status](https://travis-ci.org/spbsu-ml-community/jmll.svg?branch=master)](https://travis-ci.org/spbsu-ml-community/jmll)

### Запуск
```
java -Xmx4096m -jar broad-match.jar -depends dict-*.dict out-*.stats queries_learn.tsv.gz > stdout.txt 2> stderr.txt &
```
