#!/usr/bin/env bash

for fn in $1/train/* $1/test/*
do
  echo $fn
  mv $fn $2/
done
