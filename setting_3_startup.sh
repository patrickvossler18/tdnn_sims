#!/bin/bash
DIM=3
SETTING=setting_3.R
SETTING_NUM=3

PUB_IP=$(curl http://169.254.169.254/latest/meta-data/public-ipv4)

# install aws cli
sudo apt-get -y install awscli

curl -o /home/ubuntu/setting_1.R https://raw.githubusercontent.com/patrickvossler18/tdnn_sims/main/setting_1.R

curl -o /home/ubuntu/setting_2.R https://raw.githubusercontent.com/patrickvossler18/tdnn_sims/main/setting_2.R

curl -o /home/ubuntu/setting_3.R https://raw.githubusercontent.com/patrickvossler18/tdnn_sims/main/setting_3.R

cd /home/ubuntu
sudo -i -u ubuntu R -e "devtools::install_github('patrickvossler18/tdnn_package@rcppthread')"

sudo -i -u ubuntu Rscript $SETTING --dimension $DIM >&1 | tee "setting_${SETTING_NUM}_p_${DIM}_$(date +"%Y_%m_%d_%I_%M_%p").log"; aws s3 cp $(ls -rt -d -1 "$PWD"/setting_"${SETTING_NUM}"*.rds | tail -n 1) s3://tdnnsim/setting_3/; sudo shutdown -h now &