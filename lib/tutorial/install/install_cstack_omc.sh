#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "ERROR! Illegal number of parameters. Usage: bash install.sh conda_install_path environment_name"
    exit 0
fi

conda_install_path=$1
conda_env_name=$2

source $conda_install_path/etc/profile.d/conda.sh
echo "****************** Creating conda environment ${conda_env_name} python=3.7 ******************"
conda create -y -n $conda_env_name  python=3.7 

echo ""
echo ""
echo "****************** Activating conda environment ${conda_env_name} ******************"
conda activate $conda_env_name


echo "****************** add tsinghua source ******************"
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ 

conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/

conda config --set show_channel_urls yes


echo ""
echo ""

pip install absl-py==1.0.0
pip install attrs==21.4.0
pip install cachetools==5.0.0
pip install certifi==2021.10.8
pip install numpy==1.21.5
pip install shapely
pip install charset-normalizer==2.0.12
pip install cycler==0.11.0
pip install Cython
pip install cython-bbox==0.1.3
pip install dataclasses==0.6
pip install et-xmlfile==1.1.0
pip install flake8
pip install flake8-import-order==0.18.1
pip install fonttools==4.31.2
pip install future==0.18.2
pip install google-auth==2.6.2
pip install google-auth-oauthlib==0.4.6
pip install grpcio==1.45.0
pip install idna==3.3
pip install importlib-metadata==4.11.3
pip install iniconfig==1.1.1
pip install kiwisolver==1.4.2
pip install Markdown==3.3.6
pip install matplotlib==3.5.1
pip install mccabe==0.6.1
pip install motmetrics==1.2.0
pip install oauthlib==3.2.0
pip install opencv-python==4.5.5.64
pip install openpyxl==3.0.9
pip install packaging==21.3
pip install pandas==1.1.5
pip install Pillow==9.1.0
pip install pluggy==1.0.0
pip install protobuf==3.20.0
pip install py==1.11.0
pip install py-cpuinfo==8.0.0
pip install pyasn1==0.4.8
pip install pyasn1-modules==0.2.8
pip install pycodestyle==2.8.0
pip install pyflakes==2.4.0
pip install pyparsing==3.0.7
pip install pytest==7.1.1
pip install pytest-benchmark==3.4.1
pip install python-dateutil==2.8.2
pip install pytz==2022.1
pip install PyYAML==6.0
pip install requests==2.27.1
pip install requests-oauthlib==1.3.1
pip install rsa==4.8
pip install scipy==1.7.3
pip install seaborn==0.11.2
pip installsix==1.16.0
pip install tensorboard==2.8.0
pip install tensorboard-data-server==0.6.1
pip install tensorboard-plugin-wit==1.8.1
pip install thop==0.0.31.post2005241907
pip install tomli==2.0.1
pip install torch==1.7.0
pip install torchsummary==1.5.1
pip install torchvision==0.8.1
pip install tqdm==4.64.0
pip install typing_extensions==4.1.1
pip install urllib3==1.26.9
pip install Werkzeug==2.1.1
pip install xmltodict==0.12.0
pip install zipp==3.8.0
pip install wandb
pip install lap==0.4.0
pip install cython_bbox
pip install easydict
pip install loguru
pip install shapely

echo "****************** Installation complete! ******************"
