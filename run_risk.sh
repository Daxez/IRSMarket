#!/usr/bin/bash
exec 5>&1
aggregateid=$(pypy quick.py | tee >(cat - >&5) | sed -e '$!d')

printf "Going to do stuff with aggregateId %s", $aggregateid

python risk.py $aggregateid
