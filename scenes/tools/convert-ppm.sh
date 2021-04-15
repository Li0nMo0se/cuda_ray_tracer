#!/bin/bash
convert -depth 24 -define png:compression-filter=1  -define png:compression-level=9 -define png:compression-strategy=2 $1 $2
