# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.
ARG BASE_CONTAINER=jupyter/scipy-notebook
FROM $BASE_CONTAINER

LABEL maintainer="Jupyter Project <jupyter@googlegroups.com>"

# Install DICE
RUN pip install --upgrade https://github.com/interpretml/DiCE/tarball/master

# Install shap
RUN pip install shap