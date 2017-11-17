#!/bin/bash

wait_file() {
  local file="$1"; shift
  local wait_seconds="${1:-10}"; shift # 10 seconds as default timeout

  until test $((wait_seconds--)) -eq 0 -o -f "$file" ; do sleep 1; done

  ((++wait_seconds))
}

nohup python3 lstm_c2_generation.py $1 $2 $3 > nohup/$1.out &

# Get rid of floating tail commands from previous runs
kill $(ps a |grep 'tail -f -n 100 out' | awk '{ print $1; }')

# Wait for up to 10 seconds then commit the new config and model def
wait_file "./out/$1/config.json" && {
  touch out/$1/training.log
  NOTES_FILE=out/$1/notes.md
  if [ ! -f $NOTES_FILE ]; then
    echo "# $1 Notes" > $NOTES_FILE
    echo "" >> $NOTES_FILE
    dt=$(date '+%F %H:%M:%S'); echo "$dt" >> $NOTES_FILE
    echo "" >> $NOTES_FILE
    echo "Run with arguments $2 $3" >> $NOTES_FILE
    echo "" >> $NOTES_FILE
    echo "## Description" >> $NOTES_FILE
    echo "" >> $NOTES_FILE
  fi
  editor $NOTES_FILE
  git add out/$1/config.json
  git add out/$1/training.log
  git commit -a -m "Learning $1 $2 $3"
}

tail -f -n 100 out/$1/log &
