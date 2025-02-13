# FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime # this doesnt work on my pc
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /code

# Install system dependencies including X11 libraries and wget
RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6 libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy code folders and files
COPY ./annotator /code/annotator
COPY ./cldm /code/cldm
COPY ./share.py /code/share.py
COPY ./config.py /code/config.py
COPY ./app /code/app
COPY ./ldm /code/ldm

# Set environment variables
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
# ENV TRANSFORMERS_VERBOSITY=error


# Expose the application port
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
