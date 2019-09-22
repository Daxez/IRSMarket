#!/bin/bash
exec 5>&1
aggregateid=$(pypy quick_till_avalanche.py | tee >(cat - >&5) | sed -e '$!d')

printf "Going to do stuff with aggregateId %s", $aggregateid

python analyse_to_avalanche.py $aggregateid
