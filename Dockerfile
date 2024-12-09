FROM arm64v8/python:3.9

# Set the working directory
WORKDIR /app

# Copy all files into the container
COPY . /app

# Install system dependencies for TensorFlow and PyTorch
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    libblas-dev \
    liblapack-dev \
    gfortran

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements-2.txt

# Command to run the script
CMD ["python", "financial_sentiment-2.py"]

