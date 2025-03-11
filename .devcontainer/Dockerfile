FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /workspace

# Install build tools and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    bash-completion \
    build-essential \
    gcc \
    libffi-dev \
    python3-dev \
    pkg-config \
    libcairo2-dev \
    npm \
    default-jre \
 && rm -rf /var/lib/apt/lists/*

# Copy project requirements into the container
#COPY workspace/workspace/requirements.txt .

# Install Python dependencies
#RUN pip install --no-cache-dir -r requirements.txt

# Default entrypoint
#ENTRYPOINT ["/bin/bash"]
