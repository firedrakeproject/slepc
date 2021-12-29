#!/bin/bash
echo Commits:
git shortlog -s | sort -nr
echo Lines:
git ls-files | xargs -n1 git blame --line-porcelain HEAD |grep -ae "^author "|sort|uniq -c|sort -nr
