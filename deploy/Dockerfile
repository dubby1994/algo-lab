# 建议使用 2025 年发布的版本，以支持你的 5070 Ti
FROM nvcr.io/nvidia/tritonserver:25.01-py3

# 后面的 pip 安装建议也用最新的，因为新显卡通常需要新版 Torch 驱动
RUN pip install --no-cache-dir \
    torch \
    torchvision \
    sentence-transformers \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装其他依赖
RUN pip install --no-cache-dir sentence-transformers -i https://pypi.tuna.tsinghua.edu.cn/simple

# 设置环境变量，确保 Python 路径正确
ENV PYTHONNOUSERSITE=1