package cn.dubby.algo.lab.feature;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import java.util.Arrays;
import java.util.Collections;

public class JavaRuntimeOnnxFeatureService {

    private static final String MODEL_PATH = "triton_repo/clip_vision/1/model.onnx";

    private static final OrtEnvironment env;
    private static final OrtSession session;

    static {
        try {
            env = OrtEnvironment.getEnvironment();
            session = env.createSession(MODEL_PATH);
        } catch (Throwable t) {
            throw new RuntimeException("Failed to initialize ONNX Runtime", t);
        }
    }

    /**
     * 将1维float数组重塑为4维数组 [batch][channels][height][width]
     */
    private float[][][][] reshapeTo4D(float[][] flatData, int batch, int channels, int height, int width) {
        for (float[] singleFlatData : flatData) {
            if (singleFlatData.length != channels * height * width) {
                throw new IllegalArgumentException(
                        String.format("数据长度不匹配: 期望 %d, 实际 %d",
                                batch * channels * height * width, singleFlatData.length)
                );
            }
        }



        float[][][][] result = new float[batch][channels][height][width];

        for (int b = 0; b < batch; b++) {
            int index = 0;

            for (int c = 0; c < channels; c++) {
                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < width; w++) {
                        result[b][c][h][w] = flatData[b][index++];
                    }
                }
            }
        }
        return result;
    }

    public float[][] getVector(float[][] preprocessedData) throws OrtException {
        float[][][][] input = reshapeTo4D(preprocessedData, preprocessedData.length, 3, 224, 224);

        // 2. 创建推理环境

        // 4. 创建输入张量
        OnnxTensor inputTensor = OnnxTensor.createTensor(env, input);

        // 5. 运行推理
        String inputName = session.getInputNames().iterator().next();
        OrtSession.Result results = session.run(
                Collections.singletonMap(inputName, inputTensor)
        );

        // 6. 获取输出
        float[][] output = (float[][]) results.get(0).getValue();
        System.out.println("推理结果: " + Arrays.toString(output[0]));

        // 7. 释放资源
        inputTensor.close();

        return output;
    }

}
