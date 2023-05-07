#!/bin/bash

[ -z "$1" ] && echo "Usage: $0 <test_file>" && exit 1

cd `dirname $0`
. env/bin/activate
. env.sh
python index.py $1

