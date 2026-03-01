# 1、创建文件 Dockerfile

deploy/Dockerfile

```
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

```

# 2、执行命令

在deploy目录执行
```
docker build -t my_triton_clip .
```


# 3、创建模型文件

triton_repo/clip_vision/1/model.onnx
```
下载 https://huggingface.co/Xenova/clip-vit-base-patch32/blob/main/onnx/vision_model.onnx

```

triton_repo/clip_vision/config.pbtxt
```
name: "clip_vision"
backend: "onnxruntime"
max_batch_size: 0

input [
  {
    name: "pixel_values"
    data_type: TYPE_FP32
    dims: [ -1, 3, 224, 224 ]
  }
]

output [
  {
    name: "image_embeds"
    data_type: TYPE_FP32
    dims: [ -1, 512 ]
  }
]
```


# 4、运行docker
在 triton_repo 上一层目录执行
```
docker run --gpus all -it --rm -p 8001:8001 -v "%cd%\triton_repo:/models"  my_triton_clip tritonserver --model-repository=/models
```

# 5、运行Java代码

JVM参数
```
-Djava.library.path=C:\Code\lab\algo-lab\algo-lab\lib
```

# 其他

模型信息 https://netron.app/
