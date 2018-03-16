FROM consol/ubuntu-xfce-vnc:1.1.0

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 USER=$USER HOME=$HOME

RUN echo "The working directory is: $HOME"
RUN echo "The user is: $USER"

USER 0

RUN apt-get update && apt-get install -y \
        sudo \
        git \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# install dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    apt-utils \
    curl \
    nano \
    vim \
    git \
    zlib1g-dev \
    cmake \
    python-software-properties \
    software-properties-common


# Install python and pip
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python-numpy \
    python-dev


# Installing python3.6 and pip3
RUN add-apt-repository ppa:jonathonf/python-3.6
RUN apt update
RUN apt install -y \
    python3.6 \
    python3.6-dev \
    python3.6-venv
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python3.6 get-pip.py
RUN ln -s /usr/bin/python3.6 /usr/local/bin/python3
RUN rm get-pip.py

RUN mkdir -p $HOME
WORKDIR $HOME

# Installing Torch
RUN pip3 install http://download.pytorch.org/whl/cpu/torch-0.3.1-cp36-cp36m-linux_x86_64.whl
RUN pip3 install torchvision

# Install pip3 dependencies
#RUN git clone https://github.com/pierg/baby-ai-game.git baby-ai-game-github
#RUN cd baby-ai-game-github
#RUN pip3 install -e .
#RUN cd ..
#RUN rm -r baby-ai-game-github

# install python 3 dependencies
RUN pip3 install --upgrade pip
RUN pip3 --no-cache-dir install \
    gym>=0.9.6 \
    numpy>=1.10.0 \
    pyqt5>=5.10.1
