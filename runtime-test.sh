#!/usr/bin/env bash
 when-changed -v -r -1 -s  ./    "pytest -s -v  $1"