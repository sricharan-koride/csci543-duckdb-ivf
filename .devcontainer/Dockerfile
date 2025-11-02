# Start from a standard Ubuntu image
FROM ubuntu:22.04

# Avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install all necessary C++ build tools
# ADDED: ccache (as recommended by the DuckDB template)
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    gcc \
    cmake \
    git \
    python3 \
    python3-pip \
    ninja-build \
    ccache \
    libssl-dev \
    wget \
    && apt-get clean

# Create a non-root user for VS Code
RUN useradd -ms /bin/bash vscode
USER vscode
WORKDIR /home/vscode

# Set the default command
CMD [ "/bin/bash" ]