package cn.dubby.algo.lab;

import cn.dubby.algo.lab.dto.RankDTO;
import cn.dubby.algo.lab.feature.FeaturePreProcessService;
import cn.dubby.algo.lab.feature.TritonFeatureService;
import cn.dubby.algo.lab.util.LogUtils;
import org.opencv.core.Core;

import java.io.File;
import java.util.*;

public class Main {

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    private static final FeaturePreProcessService featurePreProcessService = new FeaturePreProcessService();
    private static final TritonFeatureService tritonFeatureService = new TritonFeatureService("localhost", 8001);

    public static void main(String[] args) {
        Map<String, float[]> map = new HashMap<>();

        File folder = new File("data/");
        File[] files = folder.listFiles();
        if (files != null) {
            for (File file : files) {
                if (!file.isDirectory()) {
                    String fileName = file.getName();
                    float[] feature = imageToFeature(file.getPath());
                    map.put(fileName, feature);
                }
            }
        }

        List<RankDTO> rankDTOList = new ArrayList<>();

        for (int i = 0; i < files.length; i++) {
            for (int j = i + 1; j < files.length; j++) {
                String leftName = files[i].getName();
                String rightName = files[j].getName();

                if (leftName.equalsIgnoreCase(rightName)) {
                    continue;
                }

                float[] leftFeature = map.get(leftName);
                float[] rightFeature = map.get(rightName);
                double similarity = dotProduct(normalize(leftFeature), normalize(rightFeature));

                RankDTO rankDTO = new RankDTO();
                rankDTO.setLeft(leftName);
                rankDTO.setRight(rightName);
                rankDTO.setSimilarity(similarity);
                rankDTOList.add(rankDTO);
            }
        }

        rankDTOList.sort(Comparator.comparingDouble(RankDTO::getSimilarity));
        for (RankDTO rankDTO : rankDTOList) {
            System.out.println(rankDTO.getLeft() + "\t" + rankDTO.getRight() + "\t" + rankDTO.getSimilarity());
        }
    }

    private static float[] imageToFeature(String imagePath) {
        float[] preprocessedFeature = featurePreProcessService.preprocess(imagePath, false);
        return tritonFeatureService.getVector("clip_vision", "pixel_values", preprocessedFeature);
    }

    public static float[] normalize(float[] v) {
        double norm = 0;
        for (float f : v) norm += f * f;
        norm = Math.sqrt(norm);
        float[] normalized = new float[v.length];
        for (int i = 0; i < v.length; i++) normalized[i] = (float) (v[i] / norm);
        return normalized;
    }

    // 归一化后的向量，点积（Dot Product）就等于余弦相似度
    public static double dotProduct(float[] v1, float[] v2) {
        double sum = 0;
        for (int i = 0; i < v1.length; i++) sum += v1[i] * v2[i];
        return sum;
    }

}
