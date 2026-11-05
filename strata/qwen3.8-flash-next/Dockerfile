# syntax=docker/dockerfile:1
FROM nvidia/cuda:13.0.0-devel-ubuntu24.04 AS build

ARG STRATA_COMMIT=fb58e0dbc8399662c0e47c76578c6e878b14f6cf
ARG CUDA_ARCHITECTURES=120
ARG BUILD_JOBS=4
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates git python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/strata
RUN git init . && git remote add origin https://github.com/Niko1221/Strata.git \
    && git fetch --depth 1 origin "$STRATA_COMMIT" && git checkout --detach FETCH_HEAD
RUN python3 -m venv .venv \
    && .venv/bin/pip install --no-cache-dir -r requirements.txt
# Fetch only the pinned ggml source, rather than llama.cpp's full git history.
RUN git init /opt/llama.cpp \
    && git -C /opt/llama.cpp remote add origin https://github.com/ggml-org/llama.cpp.git \
    && git -C /opt/llama.cpp fetch --depth 1 origin 3cf03257f219afbe7334045ff7c6a06ac68c627d \
    && git -C /opt/llama.cpp checkout --detach FETCH_HEAD
RUN .venv/bin/cmake -S . -B build -G Ninja \
    -DCMAKE_MAKE_PROGRAM=/opt/strata/.venv/bin/ninja \
    -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
    -DSTRATA_BUILD_TESTS=OFF -DSTRATA_PORTABLE=ON \
    -DSTRATA_GGML_DIR=/opt/llama.cpp \
    && .venv/bin/cmake --build build --target strata --parallel "$BUILD_JOBS"

FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 LANG=C.UTF-8
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates python3 libatomic1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/strata
COPY --from=build /opt/strata/.venv /opt/strata/.venv
COPY --from=build /opt/strata/build/strata /opt/strata/engine/strata
COPY --from=build /opt/strata/serve /opt/strata/serve
COPY --from=build /opt/strata/tools/strata_tokenizer.py /opt/strata/tools/strata_tokenizer.py
COPY --from=build /opt/strata/data/expert-profile.bin /opt/strata/data/expert-profile.bin
COPY docker/strata.json /opt/strata/strata.json
RUN mkdir -p /opt/strata/logs
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
    CMD ["/opt/strata/.venv/bin/python", "-c", "import json, urllib.request; assert json.load(urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=4))['loaded']"]
ENTRYPOINT ["/opt/strata/.venv/bin/python", "-m", "serve.server", "--engine", "strata", "--config", "/opt/strata/strata.json", "--host", "127.0.0.1", "--port", "8080"]
