#!/bin/bash

# This is a sample command. You'll probably have to replace 5200, 6200, and 100 appropriately and maybe check the file match pattern.
for v in {5200..6600..100}; do
    echo "Processing $v mV"
    echo "Voltage: $v" >> diagnostic.txt
    grep -o 1 *-${v}-* | wc -l >> diagnostic.txt
done
