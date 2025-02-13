# Using pip-based image with PyTorch 2.1.0 and CUDA 11.8
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set the working directory
WORKDIR /code

# Install system dependencies (X11 libraries, wget, etc.)
RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6 libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy project directories
COPY ./annotator /code/annotator
COPY ./cldm /code/cldm
COPY ./share.py /code/share.py
COPY ./config.py /code/config.py
COPY ./app /code/app
COPY ./ldm /code/ldm

# Set environment variables for NVIDIA GPU support
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
# (Optional) Uncomment to suppress Transformers warnings:
# ENV TRANSFORMERS_VERBOSITY=error

# Expose the application port and run the FastAPI server using uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
