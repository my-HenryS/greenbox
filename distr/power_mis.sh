#!/bin/bash

sleep 1800
for i in 5 10 15 20 25
do
    python3 simulator.py 3 80 $i > tmp/realdist_azurev1_site3_slo80_lifetimemis0_powermis${i}.txt
    python3 simulator.py 24 80 $i > tmp/realdist_azurev1_site24_slo80_lifetimemis0_powermis${i}.txt
done
