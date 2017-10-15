#!/usr/bin/env bash

for i in *.avi;
  do name=`echo $i | cut -d'.' -f1`;
  echo $name;
  ffmpeg -i "$i" "${name}.mov";
done