#!/usr/bin/env bash
cd "$(dirname "$0")/.."

for city in nyc tky ca; do
  python poi/Llama-Mob.py --city "$city"
done
