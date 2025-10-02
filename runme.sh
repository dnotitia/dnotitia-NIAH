#!/bin/bash

python run_llm_multi_needle_test.py --model-names $1 --case-names pizza_ingredients
python Needle_vis_llm.py
open ./llm_multi_needle/results/visualizations/
