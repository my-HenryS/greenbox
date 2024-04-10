sudo apt update
yes | sudo apt install python3-pip
pip install pandas
pip install matplotlib
pip install graphviz
pip install h3
pip install gurobipy
pip install joblib
pip install ray
pip install utm
wget https://packages.gurobi.com/lictools/licensetools10.0.2_linux64.tar.gz
tar zxvf licensetools10.0.2_linux64.tar.gz

./grbgetkey a306276c-25ba-421f-b8a2-395ceebb2f95
