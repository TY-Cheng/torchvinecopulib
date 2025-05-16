#!/bin/bash

PYTHON_BIN=$(which python)

for i in {0..90..10}; do
    end=$((i + 10))
    name="seeds_${i}_${end}"
    echo "Launching $name"
    nohup "$PYTHON_BIN" run_seeds.py $i $end > logs/$name.log 2>&1 &
done

wait