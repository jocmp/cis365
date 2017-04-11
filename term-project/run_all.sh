#!/usr/bin/env sh

echo 'Downloading requirements...'
pip3 install -r requirements.txt

echo
echo 'Running classifier'
python3 hive_classifier.py
