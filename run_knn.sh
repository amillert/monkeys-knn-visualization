#!/bin/bash
set -e

if [ $1 ]; then
  echo 1
  ./monkey_classify.py knn -i monkeys.csv -o $1 -d bmi fur_color_int weight
else
  echo 2
  python -m unittest -v tests.py
  ./monkey_classify.py knn -i monkeys.csv -o out.csv -d bmi fur_color_int weight
fi
