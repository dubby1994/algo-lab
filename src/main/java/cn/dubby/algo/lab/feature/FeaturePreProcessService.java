package cn.dubby.algo.lab.feature;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public class FeaturePreProcessService {

    // 默认使用 CLIP 参数
    private static final double[] CLIP_MEAN = {0.48145466, 0.4578275, 0.40821073};
    private static final double[] CLIP_STD = {0.26862954, 0.26130258, 0.27577711};

    // ResNet 标准参数
    private static final double[] IMAGENET_MEAN = {0.485, 0.456, 0.406};
    private static final double[] IMAGENET_STD = {0.229, 0.224, 0.225};

    public float[] preprocess(String imagePath, boolean isClip) {
        Mat src = Imgcodecs.imread(imagePath);
        if (src.empty()) return null;

        Mat rgb = new Mat();
        Mat resized = new Mat();
        Mat floatMat = new Mat();

        try {
            // 1. BGR -> RGB
            Imgproc.cvtColor(src, rgb, Imgproc.COLOR_BGR2RGB);

            // 2. Resize
            Imgproc.resize(rgb, resized, new Size(224, 224));

            // 3. To Float 32 & Normalize 0-1
            resized.convertTo(floatMat, CvType.CV_32FC3, 1.0 / 255.0);

            // 4. Subtract Mean & Divide Std
            double[] mean = isClip ? CLIP_MEAN : IMAGENET_MEAN;
            double[] std = isClip ? CLIP_STD : IMAGENET_STD;

            Core.subtract(floatMat, new Scalar(mean[0], mean[1], mean[2]), floatMat);
            Core.divide(floatMat, new Scalar(std[0], std[1], std[2]), floatMat);

            // 5. HWC -> NCHW
            float[] hwcData = new float[3 * 224 * 224];
            float[] nchwData = new float[3 * 224 * 224];
            floatMat.get(0, 0, hwcData);

            for (int c = 0; c < 3; c++) {
                for (int i = 0; i < 224 * 224; i++) {
                    nchwData[c * 224 * 224 + i] = hwcData[i * 3 + c];
                }
            }
            return nchwData;

        } finally {
            // 💡 长辈的叮嘱：必须释放内存，否则百万级图片清洗会撑爆系统
            src.release();
            rgb.release();
            resized.release();
            floatMat.release();
        }
    }
}