FROM continuumio/miniconda3
COPY environment.yml .
RUN conda env update --name base --file environment.yml && conda clean -a