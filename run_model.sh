#!/bin/sh

if [ "$1" = "test" ]; then
    python3 test.py "$1" "$2" "$3" "$4"
else
    python3 train.py "$1" "$2" "$3"
fi
