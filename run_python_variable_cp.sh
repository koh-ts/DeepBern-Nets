#!/bin/bash

for i in {1..10}
do
    python integrated_analysis.py --cp=2
    python integrated_analysis.py --cp=4
    python integrated_analysis.py --cp=6
    python integrated_analysis.py --cp=8
    python integrated_analysis.py --cp=10
done