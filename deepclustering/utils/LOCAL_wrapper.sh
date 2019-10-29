#!/usr/bin/env bash

local_wrapper(){
	command=$1
    echo ${command} > tmp.sh
    sed -i '1i\#!/bin/bash' tmp.sh
    bash tmp.sh
    rm tmp.sh
}