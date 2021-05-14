## Build image from tianchi base image
FROM registry.cn-shanghai.aliyuncs.com/tcc-public/python:3
FROM registry.cn-shanghai.aliyuncs.com/tcc-public/pytorch:1.7-cuda11.0-py3 
#Install gcc envs and CStrack envs
RUN apt-get -y install gcc
RUN apt install -y libgl1-mesa-glx
RUN apt-get install -y libglib2.0-dev
RUN apt-get install -y libgl1-mesa-dev
RUN pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple 
RUN pip install numpy
RUN pip install Cython -i https://pypi.tuna.tsinghua.edu.cn/simple 
RUN pip install cython-bbox  -i https://pypi.tuna.tsinghua.edu.cn/simple 
RUN pip install torchsummary==1.5.1  -i https://pypi.tuna.tsinghua.edu.cn/simple 
RUN pip install lap==0.4.0  -i https://pypi.tuna.tsinghua.edu.cn/simple 
RUN pip install motmetrics==1.2.0  -i https://pypi.tuna.tsinghua.edu.cn/simple 
RUN pip install opencv-python==4.3.0.36  -i https://pypi.tuna.tsinghua.edu.cn/simple 
RUN pip install matplotlib==3.3.0  -i https://pypi.tuna.tsinghua.edu.cn/simple 
RUN pip install tqdm==4.48.2  -i https://pypi.tuna.tsinghua.edu.cn/simple 
RUN pip install pandas==1.1.0  -i https://pypi.tuna.tsinghua.edu.cn/simple 
RUN pip install scipy==1.5.2  -i https://pypi.tuna.tsinghua.edu.cn/simple 
RUN pip install coremltools==4.1  -i https://pypi.tuna.tsinghua.edu.cn/simple 
RUN pip install onnx==1.8.1  -i https://pypi.tuna.tsinghua.edu.cn/simple 
RUN pip install Pillow==8.2.0  -i https://pypi.tuna.tsinghua.edu.cn/simple 
RUN pip install pycocotools==2.0.2  -i https://pypi.tuna.tsinghua.edu.cn/simple 
RUN pip install thop==0.0.31-2005241907  -i https://pypi.tuna.tsinghua.edu.cn/simple 
RUN apt-get install -y python-qt4
RUN apt-get  install zip  unzip
RUN conda install -c anaconda mpi4py --yes  
RUN pip install seaborn -i https://pypi.tuna.tsinghua.edu.cn/simple 
RUN pip install ensemble-boxes -i https://pypi.tuna.tsinghua.edu.cn/simple 
RUN pip install tensorboard -i https://pypi.tuna.tsinghua.edu.cn/simple
## Build the files in the current folder into the root directory of the image
ADD . /
## Specify the default working directory as the root directory (you need to put both test_panda.sh and the generated result files to this folder)
WORKDIR /
## test_panda.sh will be executed after the image initialization
CMD ["sh", "test_panda.sh"]