#!/bin/bash

CLOUDUTIL=90

overall(){
    for starttime in 0 7 14 21 28 35 42 49
    do
        for slo in 90
        do
            for dist in 10
            do
                python3 simulator.py $1 $slo 0 0 $dist $starttime 5 &> tmp/raw/realdist_azurev1_site$1_slo${slo}_lifetimemis0_powermis0_dist${dist}_starttime${starttime}_util${CLOUDUTIL}.txt &
            done
        done

        python3 simulator.py $1 100 0 0 0 $starttime 5 &> tmp/raw/realdist_azurev1_site$1_slo100_lifetimemis0_powermis0_dist0_startime${starttime}_util${CLOUDUTIL}.txt &
    done
}

normal(){
    for k in 42
    do
        for i in 90
        do
            for j in 10 20 30 40 50
            do
                python3 simulator.py $1 $i 0 0 $j $k 5 &> tmp/raw/realdist_azurev1_site$1_slo${i}_lifetimemis0_powermis0_dist${j}_starttime${k}_centralized.txt &
            done
        done

        python3 simulator.py $1 100 0 0 0 $k 5 &> tmp/raw/realdist_azurev1_site$1_slo100_lifetimemis0_powermis0_dist0_startime${k}_centralized.txt &
    done
}

slo(){
    for k in 42
    do
        for i in 70 80 95
        do
            for j in 10
            do
                python3 simulator.py $1 $i 0 0 $j $k 5 &> tmp/raw/realdist_azurev1_site$1_slo${i}_lifetimemis0_powermis0_dist${j}_starttime${k}_centralized.txt &
            done
        done

    done
}

test (){
    for i in 49
    do
        python3 simulator.py $1 90 0 0 10 $i 4 &> tmp/raw/realdist_azurev1_site$1_slo90_lifetimemis0_powermis0_dist10_starttime${i}_newsites.txt &
    done
}

test2 (){
    python3 simulator.py $1 90 0 0 10 0 &> tmp/raw/test.txt &
}

lahead_test() {
# for i in 0 7
for i in 7 
    do
        for k in 2 3 4 5 6 7
        # for k in 1
        do
            for j in 90
            do
                for z in 50
                # for z in 0
                do
                    python3 simulator.py $1 $j 0 0 $z $i $k &> tmp/raw/realdist_azurev1_site$1_slo${j}_lifetimemis0_powermis0_dist${z}_starttime${i}_lookahead${k}_newsites.txt &
                done
            done
        done
    done
}

power_mispred() {
for i in 0 1 2 3 4 5 6
do
    for k in 4
    do
        for j in 90
        do
            for z in 10
            do
                for m in -15 -5 5 15
                # for m in 40
                do
                python3 simulator.py $1 $j $m 0 $z $i $k &> tmp/raw/realdist_azurev1_site$1_slo${j}_lifetimemis0_neg_powermis${m}_dist${z}_starttime${i}_lookahead${k}_newsites_util90.txt &
                done
            done
        done
    done
done
}

vmlifetime_mispred() {
for i in 7
do
    for k in 3
    do
        for j in 100
        do
            for z in 10 20 30 40
            do
                for m in 5 15
                do
                python3 simulator.py $1 $j $m 0 $z $i $k &> tmp/raw/realdist_azurev1_site$1_slo${j}_lifetimemis0_powermis${m}_dist${z}_starttime${i}_lookahead${k}_newsites.txt &
                done
            done
        done
    done
done
}

scale(){
    for k in 7
    do
        for i in 90
        do
            for j in 10
            do
                python3 simulator.py $1 $i 0 0 $j $k 4 &> tmp/raw/realdist_azurev1_site$1_slo${i}_lifetimemis0_powermis0_dist${j}_starttime${k}_centralized.txt &
            done
        done
    done
}

# normal "$1"
# test "$1"
# lahead_test "$1"
# power_mispred "$1"
$2 "$1"

# for i in 70
# do
#     for j in 10 20 30 40 50
#     do
#         python3 simulator.py $1 $i 0 0 $j &> tmp/raw/realdist_azurev1_site$1_slo${i}_lifetimemis0_powermis0_dist${j}_centralized.txt &
#     done
# done


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
