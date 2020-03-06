FROM tensorflow/tensorflow:latest-py3
RUN pip install jupyterlab
RUN pip install matplotlib
WORKDIR /notebooks
ENTRYPOINT ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
