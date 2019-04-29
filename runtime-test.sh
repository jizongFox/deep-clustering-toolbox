#!/usr/bin/env bash
 when-changed -v -r -1 -s ./    "python -m pytest -s $1"