package com.expleague.exp.nn;

import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.random.FastRandom;
import com.expleague.commons.seq.IntSeq;
import com.expleague.ml.data.set.impl.VecDataSetImpl;
import com.expleague.ml.loss.blockwise.BlockwiseMLLLogit;
import com.expleague.ml.methods.nn.NeuralTreesOptimization;
import com.expleague.ml.models.nn.ConvNet;
import com.expleague.ml.models.nn.NetworkBuilder;
import com.expleague.ml.models.nn.layers.*;
import com.expleague.nn.MNISTUtils;
import org.junit.Before;
import org.junit.Test;

import static com.expleague.nn.MNISTUtils.*;

public class LeTreeNetExperiments {
  private static final String PATH_TO_LENET_MODEL = "/Users/solar/Downloads/lenet.nn";
  private static final FastRandom rng = new FastRandom();
  private static Mx trainSamples = new VecBasedMx(numTrainSamples, MNISTUtils.widthIn * MNISTUtils.heightIn);
  private static Mx testSamples = new VecBasedMx(numTestSamples, MNISTUtils.widthIn * MNISTUtils.heightIn);
  private static int[] trainLabels = new int[numTrainSamples];
  private static int[] testLabels = new int[numTestSamples];
  private static BlockwiseMLLLogit loss;
  private VecDataSetImpl learn;

  @Before
  public void setUp() {
    readMnist(trainLabels, trainSamples, testLabels, testSamples);
    learn = new VecDataSetImpl(trainSamples, null);
    loss = new BlockwiseMLLLogit(new IntSeq(trainLabels), learn);
  }

  private static ConvNet createConvLeNet() {
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
            .build(new OneOutLayer());

    System.out.println(network);

    return new ConvNet(network);
  }

  private static ConvNet createMobileNet() {
    final NetworkBuilder<Vec>.Network network =
        new NetworkBuilder<>(new ConstSizeInput3D(heightIn, widthIn, 1))
            .append(ConvLayerBuilder.create()
                .channels(16)
                .ksize(3,3)
                .weightFill(FillerType.XAVIER))
            .append(PoolLayerBuilder.create().ksize(3, 3).stride(3, 3))
            .append(ConvLayerBuilder.create()
                .channels(32)
                .ksize(3, 3)
                .weightFill(FillerType.XAVIER))
            .append(PoolLayerBuilder.create().ksize(2, 2).stride(2, 2))
            .build(new OneOutLayer());

    System.out.println(network);

    return new ConvNet(network);
  }

  @Test
  public void trainOnLeNetFeatures() {
    final ConvNet nn = createConvLeNet();
    nn.load(PATH_TO_LENET_MODEL, 4);

    NeuralTreesOptimization optimization =
        new NeuralTreesOptimization(100, 20000, 100, 64, 1e-3,
            1000, 4, nn, rng, System.out);
    optimization.setTest(new VecDataSetImpl(testSamples, null), new IntSeq(testLabels));
    optimization.fit(learn, loss);
  }

  @Test
  public void trainSgdOnLeNetFeatures() {
    final ConvNet nn = createConvLeNet();
    nn.load(PATH_TO_LENET_MODEL, 4);

    NeuralTreesOptimization optimization =
        new NeuralTreesOptimization(1000, 512, 500, 64, 1e-5,
            15000, 0.8, nn, rng, System.out);
    optimization.setTest(new VecDataSetImpl(testSamples, null), new IntSeq(testLabels));
    optimization.fit(learn, loss);
  }

  @Test
  public void trainMobileNetFromScratch() {
    final ConvNet nn = createMobileNet();

    NeuralTreesOptimization optimization =
        new NeuralTreesOptimization(1000, 4000, 300, 64, 1e-8,
            3000, 0.3, nn, rng, System.out);
    optimization.setTest(new VecDataSetImpl(testSamples, null), new IntSeq(testLabels));
    optimization.fit(learn, loss);
  }
}
