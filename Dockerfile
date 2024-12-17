FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

WORKDIR /app

RUN pip3 install numpy pillow h5py tqdm matplotlib pymoo 

CMD ["/bin/bash"]