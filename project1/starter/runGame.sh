#!/bin/bash

if [ -f *.hlt ]; then
    rm *.hlt
fi

if [ -f *.log ]; then
    rm *.log
fi

current_bot="../MyBot.py"
opponent_bot="OverkillBot.py"

if hash python3 2>/dev/null; then
    ./halite -d "44 44" "python3 $current_bot" "python3 $opponent_bot"
else
    ./halite -d "44 44" "python $current_bot" "python $opponent_bot"
fi
