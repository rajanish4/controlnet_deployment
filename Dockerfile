# for conda:
# # FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime
# FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# WORKDIR /code

# # Install system dependencies including X11 libraries
# # RUN apt-get update && apt-get install -y \
# #     libgl1-mesa-glx \
# #     libglib2.0-0 \
# #     libsm6 \
# #     libxext6 \
# #     libxrender-dev \
# #     wget \
# #     && rm -rf /var/lib/apt/lists/*
# RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6 wget  && rm -rf /var/lib/apt/lists/*

# # Install Miniconda
# RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
#     && bash miniconda.sh -b -p /opt/conda -u \
#     && rm miniconda.sh

# # Add conda to path
# ENV PATH=/opt/conda/bin:$PATH

# # Copy environment.yaml
# COPY environment.yaml .

# # Create conda environment
# RUN conda env create -f environment.yaml

# # Make RUN commands use the new environment
# SHELL ["conda", "run", "-n", "control_fastapi", "/bin/bash", "-c"]

# # Copy the necessary ControlNet folders and files
# COPY ./annotator /code/annotator
# COPY ./cldm /code/cldm
# COPY ./share.py /code/share.py
# COPY ./config.py /code/config.py
# COPY ./app /code/app
# COPY ./ldm /code/ldm

# # Set environment variables for CUDA
# ENV NVIDIA_VISIBLE_DEVICES=all
# ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# # Set the default command to run the FastAPI app
# CMD ["conda", "run", "--no-capture-output", "-n", "control_fastapi", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]


# for pip:
# FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime # this doesnt work on my pc
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /code

# Install system dependencies including X11 libraries and wget
RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6 libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy your code folders and files
COPY ./annotator /code/annotator
COPY ./cldm /code/cldm
COPY ./share.py /code/share.py
COPY ./config.py /code/config.py
COPY ./app /code/app
COPY ./ldm /code/ldm

# Set environment variables
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV TRANSFORMERS_VERBOSITY=error


# Expose the application port and set the default command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
