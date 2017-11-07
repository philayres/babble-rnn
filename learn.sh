#!/bin/bash

wait_file() {
  local file="$1"; shift
  local wait_seconds="${1:-10}"; shift # 10 seconds as default timeout

  until test $((wait_seconds--)) -eq 0 -o -f "$file" ; do sleep 1; done

  ((++wait_seconds))
}

nohup python3 lstm_c2_generation.py $1 $2 $3 > nohup/$1.out &
# Wait for up to 10 seconds then commit the new config and model def
wait_file "./out/$1/config.json" && {
  git add out/$1/config.json
  git commit -a -m "Learning $1 $2 $3"
}
