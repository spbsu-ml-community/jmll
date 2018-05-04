package com.expleague.nn;

import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.random.FastRandom;
import com.expleague.ml.data.set.impl.VecDataSetImpl;
import com.expleague.ml.loss.CrossEntropy;
import com.expleague.ml.methods.nn.NeuralTreesOptimization;
import com.expleague.ml.models.nn.ConvNet;
import com.expleague.ml.models.nn.NetworkBuilder;
import com.expleague.ml.models.nn.layers.*;

import java.util.function.Function;

import static com.expleague.nn.MNISTUtils.*;

public class LeTreeNet {
  private static final String pathToModel = "experiments/src/main/resources/letreenet.nn";
  private static final ConvNet leNet = createLeNet();
  private static final FastRandom rng = new FastRandom();
  private static Mx trainSamples = new VecBasedMx(numTrainSamples, MNISTUtils.widthIn * MNISTUtils.heightIn);
  private static Mx testSamples = new VecBasedMx(numTestSamples, MNISTUtils.widthIn * MNISTUtils.heightIn);
  private static double[] trainLabels = new double[numTrainSamples];
  private static double[] testLabels = new double[numTestSamples];
  private static CrossEntropy loss;

  public static void main(String[] args) {
    readMnist(trainLabels, trainSamples, testLabels, testSamples);
    train();
  }

  private static ConvNet createLeNet() {
    final NetworkBuilder<Vec>.Network network =
        new NetworkBuilder<>(new ConstSizeInput3D(heightIn, widthIn, 1))
            .append(ConvLayerBuilder.create()
                .channels(20)
                .ksize(5,5)
                .weightFill(FillerType.XAVIER))
            .append(PoolLayerBuilder.create().ksize(2, 2).stride(2, 2))
            .append(ConvLayerBuilder.create()
                .channels(50)
                .ksize(5, 5)
                .weightFill(FillerType.XAVIER))
            .append(PoolLayerBuilder.create().ksize(2, 2).stride(2, 2))
            .append(FCLayerBuilder.create().nOut(400).weightFill(FillerType.XAVIER))
            .build(new OneOutLayer());

    System.out.println(network);

    return new ConvNet(network);
  }

  public static void train() {
    final Vec weights = new ArrayVec(trainSamples.rows());
    VecTools.fill(weights, 1.);

    loss = new CrossEntropy(new ArrayVec(trainLabels),
        new VecDataSetImpl(trainSamples, null), nClasses);

    NeuralTreesOptimization<CrossEntropy> optimization =
        new NeuralTreesOptimization<>(100, 10000, leNet, rng);
    final Function<Vec, Vec> leTreeNet = optimization.fit(new VecDataSetImpl(trainSamples, null), loss);

    testModel(leTreeNet, trainLabels, trainSamples, testLabels, testSamples);
  }
}
