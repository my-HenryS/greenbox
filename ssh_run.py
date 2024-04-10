import time
import subprocess
import json
import sys

if len(sys.argv) >= 2:
    mode = sys.argv[1]
    if len(sys.argv) >= 3:
        cluster = int(sys.argv[2])
    else:
        cluster = 0
        print("please specify cluster index: 1->12nodes 2->15ndoes\n")
else:
    mode = "normal"
    cluster = 0
        
username = "zibo"
hosts = ["011114", "011110", "011105", "011106", "011104", "011018", "011109", "011014", "011108", "011020", "011112", "011101"]
hosts += ["030832", "011307", "011129", "030831", "011124", "011123", "011117", "011111", "011309", "011102"]
hosts = hosts[0:]
host_addrs = [f"{username}@c220g2-{host}.wisc.cloudlab.us" for host in hosts]
# hosts2 = ["030602", "030822", "031129", "030814", "030621", "030615", "030603", "030811", "030810", "030812", "030806", "030610", "031122", "030623", "030609"]
hosts2 = ["030602", "030822", "031129", "030814", "030621"]
host_addrs2 = [f"{username}@c220g1-{host}.wisc.cloudlab.us" for host in hosts2]
hosts_all = hosts + hosts2
host_addrs_all = host_addrs + host_addrs2

if cluster == 1:
    for i in range(len(hosts)):
        if mode == "clean":
            src_retval = subprocess.Popen(f"ssh -t {host_addrs[i]} 'killall -9 python3' ", shell=True)
            src_retval = subprocess.Popen(f"ssh -t {host_addrs[i]} 'sudo rm -rf /tmp/ray/*' ", shell=True)
        else:
            src_retval = subprocess.Popen(f"ssh -t {host_addrs[i]} 'cd /proj/cxlssdsimulator-PG0/virtual-battery-20; python3 expt.py {i} {mode} {cluster}; sleep 10; sync' ", shell=True)
elif cluster == 2:
    for i in range(len(hosts2)):
        if mode == "clean":
            src_retval = subprocess.Popen(f"ssh -t {host_addrs2[i]} 'killall -9 python3' ", shell=True)
            src_retval = subprocess.Popen(f"ssh -t {host_addrs2[i]} 'sudo rm -rf /tmp/ray/*' ", shell=True)
        else:
            src_retval = subprocess.Popen(f"ssh -t {host_addrs2[i]} 'cd /proj/cxlssdsimulator-PG0/virtual-battery-20; python3 expt.py {i} {mode} {cluster}; sleep 10; sync' ", shell=True)
elif cluster == 3:
    for i in range(len(hosts_all)):
        if mode == "clean":
            src_retval = subprocess.Popen(f"ssh -t {host_addrs_all[i]} 'killall -9 python3' ", shell=True)
            src_retval = subprocess.Popen(f"ssh -t {host_addrs_all[i]} 'sudo rm -rf /tmp/ray/*' ", shell=True)
        else:
            src_retval = subprocess.Popen(f"ssh -t {host_addrs_all[i]} 'cd /proj/cxlssdsimulator-PG0/virtual-battery-20; python3 expt.py {i} {mode} {cluster}; sleep 10; sync' ", shell=True)
