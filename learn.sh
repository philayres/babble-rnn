#!/bin/bash
nohup python lstm_c2_generation.py $1 $2 $3 > nohup-$1.out &
