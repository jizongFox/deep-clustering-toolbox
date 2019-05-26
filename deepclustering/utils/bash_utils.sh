#!/usr/bin/env bash

wait_script(){
FAIL=0
for job in `jobs -p`
do
#echo $job
    wait $job || let "FAIL+=1"
done
#echo $FAIL
if [ "$FAIL" == "0" ];
then
echo "waiting ends"
else
echo "FAIL! ($FAIL)"
fi
}