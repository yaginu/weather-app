FROM gcr.io/deeplearning-platform-release/base-cpu
RUN pip install -U fire tensorflow==2.5.0 cloudml-hypertune tensorflow-transform==1.1.0 tensorflow-addons
WORKDIR /app
COPY ["train.py", "create_dataset.py", "create_model.py", "save_model.py", "./"]

ENTRYPOINT ["python", "train.py"]