package com.expleague.nn;

import com.expleague.commons.math.FuncC1;
import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.random.FastRandom;
import com.expleague.commons.seq.IntSeq;
import com.expleague.ml.data.set.impl.VecDataSetImpl;
import com.expleague.ml.func.FuncEnsemble;
import com.expleague.ml.func.generic.ReLU;
import com.expleague.ml.loss.CrossEntropy;
import com.expleague.ml.models.nn.ConvNet;
import com.expleague.ml.models.nn.NetworkBuilder;
import com.expleague.ml.models.nn.layers.*;
import com.expleague.ml.optimization.Optimize;
import com.expleague.ml.optimization.impl.AdamDescent;

import static com.expleague.nn.MNISTUtils.*;

public class LeNetModel {
  private static final String pathToModel = "experiments/src/main/resources/lenet.nn";
  private static final ConvNet leNet = createLeNet();
  private static final FastRandom rng = new FastRandom();
  private static Mx trainSamples = new VecBasedMx(numTrainSamples, MNISTUtils.widthIn * MNISTUtils.heightIn);
  private static Mx testSamples = new VecBasedMx(numTestSamples, MNISTUtils.widthIn * MNISTUtils.heightIn);
  private static int[] trainLabels = new int[numTrainSamples];
  private static int[] testLabels = new int[numTestSamples];
  private static CrossEntropy loss;

  public static void main(String[] args) {
    readMnist(trainLabels, trainSamples, testLabels, testSamples);
    train();
  }

  private static class ConvNetSample extends FuncC1.Stub {
    private final Vec argument;
    private final FuncC1 loss;

    ConvNetSample(int idx) {
      argument = trainSamples.row(idx);
      loss = LeNetModel.loss.block(idx);
    }

    @Override
    public Vec gradientTo(Vec weights, Vec to) {
      return leNet.gradientTo(argument, weights, loss, to);
    }

    @Override
    public double value(Vec weights) {
      return loss.value(leNet.apply(argument, weights));
    }

    @Override
    public int dim() {
      return nClasses;
    }
  }

  private static void train() {
    final Vec weights = new ArrayVec(trainSamples.rows());
    VecTools.fill(weights, 1.);

    loss = new CrossEntropy(new IntSeq(trainLabels),
        new VecDataSetImpl(trainSamples, null), nClasses);

    final ConvNetSample[] funcs = new ConvNetSample[trainSamples.rows()];
    for (int i = 0; i < funcs.length; i++) {
      funcs[i] = new ConvNetSample(i);
    }

    final FuncEnsemble<FuncC1> func = new FuncEnsemble<>(funcs, weights);
    final Optimize<FuncEnsemble<? extends FuncC1>> optimizer =
        new AdamDescent(rng, 20, 32, 1e-3);

    optimizer.optimize(func, leNet.weights());

    testSoftmaxModel(leNet, trainLabels, trainSamples, testLabels, testSamples);

    leNet.save(pathToModel);
  }

  private static ConvNet createLeNet() {
    NetworkBuilder<Vec>.Network network =
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
        .append(FCLayerBuilder.create().nOut(500).weightFill(FillerType.XAVIER).activation(ReLU.class))
        .append(FCLayerBuilder.create().nOut(10).weightFill(FillerType.XAVIER))
        .build(new OneOutLayer());

    System.out.println(network);

    return new ConvNet(network);
  }
}
