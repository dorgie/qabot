#!/bin/bash

cd `dirname $0`
. env/bin/activate
. env.sh
python bot.py

