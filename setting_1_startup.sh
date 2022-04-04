#!/bin/bash
DATA=normal
SETTING=setting_1.R
SETTING_NUM=1

PUB_IP=$(curl http://169.254.169.254/latest/meta-data/public-ipv4)

curl -o /home/ubuntu/setting_1.R https://raw.githubusercontent.com/patrickvossler18/tdnn/master/non_parametric_sims/setting_1/setting_1.R?token=GHSAT0AAAAAABNXRMZQA6UPHNPD6KTYQIFOYSQQC2Q

curl -o /home/ubuntu/setting_2.R https://raw.githubusercontent.com/patrickvossler18/tdnn/master/non_parametric_sims/setting_2/setting_2.R?token=GHSAT0AAAAAABNXRMZRYPTCHROAAO36FJNOYSQQDZQ

curl -o /home/ubuntu/setting_3.R https://raw.githubusercontent.com/patrickvossler18/tdnn/master/non_parametric_sims/setting_3/setting_3.R?token=GHSAT0AAAAAABNXRMZRVHZC3G2PIRCVPZSEYSQQEIQ

cd /home/ubuntu
sudo -i -u ubuntu R -e "devtools::install_github('patrickvossler18/tdnn_package@rcppthread')"

sudo -i -u ubuntu Rscript $SETTING --data_type $DATA >&1 | tee "setting_${SETTING_NUM}_p_${DATA}_$(date +"%Y_%m_%d_%I_%M_%p").log"; sudo shutdown -h now &

