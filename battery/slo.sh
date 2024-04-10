#!/bin/bash

normal(){
    for k in 7
    do
        for i in 90
        do
            for j in 10 
            do
                # python3 simulator.py $1 $i 0 0 $j $k &> tmp/raw/realdist_azurev1_site$1_slo${i}_lifetimemis0_powermis0_dist${j}_starttime${k}_battery.txt &
                python3 simulator.py $1 $i 0 0 $j $k &> tmp/raw/realdist_azurev1_site$1_slo${i}_lifetimemis0_powermis0_dist${j}_starttime${k}_battery_util90.txt &
            done
        done

        # python3 simulator.py $1 100 0 0 0 $k &> tmp/raw/realdist_azurev1_site$1_slo100_lifetimemis0_powermis0_dist0_starttime${k}_battery.txt &
        python3 simulator.py $1 100 0 0 0 $k &> tmp/raw/realdist_azurev1_site$1_slo100_lifetimemis0_powermis0_dist0_starttime${k}_battery_util90.txt &
    done
}

slo(){
    for k in 42
    do
        for i in 70 80 95
        do
            for j in 10
            do
                python3 simulator.py $1 $i 0 0 $j $k &> tmp/raw/realdist_azurev1_site$1_slo${i}_lifetimemis0_powermis0_dist${j}_starttime${k}_battery.txt &
            done
        done

    done
}

test (){
    for i in 0 7 14 21 28 35 42 49
    do
        python3 simulator.py $1 90 0 0 10 $i &> tmp/raw/realdist_azurev1_site$1_slo90_lifetimemis0_powermis0_dist10_starttime${i}_battery.txt &
    done
}

test2 (){
    python3 simulator.py $1 90 0 0 10 0 &> tmp/raw/test.txt &
}

scale(){
    for k in 7
    do
        for i in 90
        do
            for j in 10
            do
                python3 simulator.py $1 $i 0 0 $j $k 4 &> tmp/raw/realdist_azurev1_site$1_slo${i}_lifetimemis0_powermis0_dist${j}_starttime${k}_battery.txt &
            done
        done
    done
}

# normal "$1"
# test "$1"
$2 "$1"


#for i in 80 90 100
#do
#	for j in -20 -10 10 20
#	do
#    	python3 simulator.py 3 $i $j 0 > tmp/realdist_azurev1_site3_slo${i}_lifetimemis0_powermis${j}.txt
#	done
#done

#for i in 80 90 100
#do
#	for j in -20 -10 10 20
#	do
#    	python3 simulator.py 3 $i 0 $j> tmp/realdist_azurev1_site3_slo${i}_lifetimemis${j}_powermis0.txt
#	done
#done

# for i in 70 80 90
# do
# 	#for j in 20 40 60 80 100
# 	for j in 0
# 	do
#     	python3 simulator.py 3 $i 0 0 $j > tmp/realdist_azurev1_site3_slo${i}_lifetimemis0_powermis0_dist${j}.txt
# 	done
# done

#for i in 70 80 90 100
#do
#	for j in -30 -20 -10 10 20 30
#	do
#   	python3 simulator.py 24 $i $j 0 0 > tmp/realdist_azurev1_site24_slo${i}_lifetimemis0_powermis${j}.txt
#	done
#done
