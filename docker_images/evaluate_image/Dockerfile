FROM gcr.io/deeplearning-platform-release/base-cpu
RUN pip install -U fire tensorflow==2.5.0 tensorflow-transform==1.1.0 kfp==1.6.6 

COPY ["setup.py", "create_dataset.py", "get_default.py", "./"]
RUN python setup.py install