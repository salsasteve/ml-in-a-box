#!/bin/bash

# Step 1: Start Docker container
docker run -d --gpus all --rm --name triton-server-container -v /path/to/models:/models nvcr.io/nvidia/tritonserver:23.08-py3

# Step 2: Wait for the container to be fully up
echo "Waiting for Docker container to initialize..."
sleep 10  # Adjust based on initialization time

# Step 3: Start Triton Inference Server inside the container
docker exec -it triton-server-container /bin/bash -c "/apps/start-triton-server.sh --models yolov9-c,yolov7 --model_mode eval --efficient_nms enable --opt_batch_size 4 --max_batch_size 4 --instance_group 1"

echo "Triton Inference Server started."
