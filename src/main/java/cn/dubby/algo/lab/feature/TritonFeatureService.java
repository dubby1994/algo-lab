package cn.dubby.algo.lab.feature;

import com.google.protobuf.ByteString;
import inference.GRPCInferenceServiceGrpc;
import inference.GrpcService;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;

public class TritonFeatureService {

    private final GRPCInferenceServiceGrpc.GRPCInferenceServiceBlockingStub stub;

    public TritonFeatureService(String host, int port) {
        // 1. 建立通道
        ManagedChannel channel = ManagedChannelBuilder.forAddress(host, port)
                .usePlaintext() // 本地不加密
                .build();
        // 2. 创建阻塞客户端
        this.stub = GRPCInferenceServiceGrpc.newBlockingStub(channel);
    }

    public float[] getVector(String modelName, String inputName, float[] preprocessedData) {
        // 1. 构建输入 Tensor
        GrpcService.ModelInferRequest.InferInputTensor.Builder inputTensor =
                GrpcService.ModelInferRequest.InferInputTensor.newBuilder()
                        .setName(inputName) // 传入 "main" 或 "IMAGE_INPUT"
                        .setDatatype("FP32")
                        .addShape(1).addShape(3).addShape(224).addShape(224);

        // 2. 写入数据
        for (float val : preprocessedData) {
            inputTensor.getContentsBuilder().addFp32Contents(val);
        }

        // 3. 组装请求
        GrpcService.ModelInferRequest request = GrpcService.ModelInferRequest.newBuilder()
                .setModelName(modelName)
                .addInputs(inputTensor)
                .build();

        // 4. 发送请求
        GrpcService.ModelInferResponse response = stub.modelInfer(request);

        // 💡 重点：Triton 的响应有两种存放方式
        // 方式 A: 在 RawOutputContents 里（二进制，性能高）
        // 方式 B: 在 Fp32Contents 里（你目前用的，方便但略慢）
        // 如果 Fp32Contents 为空，说明数据在 RawOutputContents 里，需要用 ByteBuffer 解析
        List<Float> resultList = response.getOutputs(0).getContents().getFp32ContentsList();

        // 容错处理：如果 Protobuf 没直接返回列表，说明你需要处理 raw 字节流
        if (resultList.isEmpty()) {
            // 这里先预警，如果跑出来全 0，说明要换成 ByteBuffer 解析方式
            // 2. 方式 A: 处理 RawOutputContents (ONNX 常走这条路)
            // 💡 重点：Raw 内容存放在 response 级别的 getRawOutputContents(index) 中
            // 而不是在 output 节点内部
            ByteString rawData = response.getRawOutputContents(0);
            ByteBuffer buffer = rawData.asReadOnlyByteBuffer().order(ByteOrder.LITTLE_ENDIAN);

            // 计算 float 数量 (每个 float 占 4 字节)
            int floatCount = rawData.size() / 4;
            float[]  vector = new float[floatCount];

            for (int i = 0; i < floatCount; i++) {
                vector[i] = buffer.getFloat();
            }

            return vector;
        }

        float[] vector = new float[resultList.size()];
        for (int i = 0; i < resultList.size(); i++) {
            vector[i] = resultList.get(i);
        }
        return vector;
    }
}