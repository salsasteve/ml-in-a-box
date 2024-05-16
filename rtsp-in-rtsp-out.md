```mermaid
graph LR
    A[Camera (RTSP)] -->|Stream| B[DeepStream Server]
    B -->|Inference Request| C[Triton Inference Server]
    C -->|Inference Response| B
    B -->|Output Stream| D[RTSP Server]
```

This diagram illustrates the following flow:
1. The camera streams video via RTSP to the DeepStream server.
2. The DeepStream server sends inference requests to the Triton Inference Server.
3. The Triton Inference Server processes the requests and sends the inference results back to the DeepStream server.
4. The DeepStream server outputs the processed video stream to an RTSP server.
