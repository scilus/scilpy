FROM scilus/scilus:latest

RUN groupadd -g 1001 scilus && \
    useradd -s /bin/bash -m -u 1001 -g scilus scilus

USER root

RUN apt update && apt install -y \
        git \
        wget \
    && rm -rf /var/lib/apt/lists/*

USER scilus
ENV PATH=/home/scilus/.local/bin:$PATH
