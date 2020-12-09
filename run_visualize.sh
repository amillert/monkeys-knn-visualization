#!/bin/bash
set -e

if [ $1 ]; then
  if [ -e $1 ]; then
    echo 1
    python -m unittest -v tests.py
    ./monkey_classify.py visual -i $1 -f weight fur_color_int
  else
    echo 2
    python -m unittest -v tests.py
    ./run_knn.sh $1 && ./monkey_classify.py visual -i $1 -f weight fur_color_int
  fi
else
  echo 3
  python -m unittest -v tests.py
  ./run_knn.sh out.csv && ./monkey_classify.py visual -i out.csv -f weight fur_color_int
fi
