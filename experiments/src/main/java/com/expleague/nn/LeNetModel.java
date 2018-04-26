package com.expleague.nn;

import com.expleague.commons.math.FuncC1;
import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.random.FastRandom;
import com.expleague.ml.data.set.impl.VecDataSetImpl;
import com.expleague.ml.func.FuncEnsemble;
import com.expleague.ml.func.generic.LogSoftMax;
import com.expleague.ml.func.generic.ReLU;
import com.expleague.ml.loss.CrossEntropy;
import com.expleague.ml.models.nn.ConvNet;
import com.expleague.ml.models.nn.NetworkBuilder;
import com.expleague.ml.models.nn.layers.*;
import com.expleague.ml.optimization.Optimize;
import com.expleague.ml.optimization.impl.AdamDescent;

import java.io.IOException;
import java.io.InputStream;
import java.net.MalformedURLException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.Arrays;
import java.util.function.Consumer;

public class LeNetModel {
  private static final String urlTrain = "https://pjreddie.com/media/files/mnist_train.csv";
  private static final String urlTest = "https://pjreddie.com/media/files/mnist_test.csv";
  private static final String pathTrain = "experiments/src/main/resources/mnist_train.csv";
  private static final String pathTest = "experiments/src/main/resources/mnist_test.csv";
  private static final String pathToModel = "experiments/src/main/resources/lenet.nn";
  private static final ConvNet leNet = createLeNet();
  private static final FastRandom rng = new FastRandom();
  private static final int heightIn = 28;
  private static final int widthIn = 28;
  private static final int nClasses = 10;
  private static Mx testSamples;
  private static Mx trainSamples;
  private static double[] trainLabels;
  private static double[] testLabels;
  private static CrossEntropy loss;

  public static void main(String[] args) {
    downloadFile(urlTrain, pathTrain);
    downloadFile(urlTest, pathTest);
    readMnist();
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

    loss = new CrossEntropy(new ArrayVec(trainLabels),
        new VecDataSetImpl(trainSamples, null), nClasses);

    final ConvNetSample[] funcs = new ConvNetSample[trainSamples.rows()];
    for (int i = 0; i < funcs.length; i++) {
      funcs[i] = new ConvNetSample(i);
    }

    final FuncEnsemble<FuncC1> func = new FuncEnsemble<>(funcs, weights);
    final Optimize<FuncEnsemble<? extends FuncC1>> optimizer =
        new AdamDescent(rng, 20, 32, 1e-3);

    optimizer.optimize(func, leNet.weights());

    int trainCounter = 0;
    for (int i = 0; i < trainLabels.length; i++) {
      Vec result = leNet.apply(trainSamples.row(i));
      final double sumExp = LogSoftMax.sumExp(result);
      double max = Double.NEGATIVE_INFINITY;
      int max_idx = -1;
      for (int j = 0; j < result.dim(); j++) {
        double value = Math.exp(result.get(j)) / sumExp;
        if (value > max) {
          max = value;
          max_idx = j;
        }
      }

      if (max_idx == trainLabels[i]) {
        trainCounter++;
      }
    }

    int testCounter = 0;
    for (int i = 0; i < testLabels.length; i++) {
      Vec result = leNet.apply(testSamples.row(i));
      final double sumExp = LogSoftMax.sumExp(result);
      double max = Double.NEGATIVE_INFINITY;
      int max_idx = -1;
      for (int j = 0; j < result.dim(); j++) {
        double value = Math.exp(result.get(j)) / sumExp;
        if (value > max) {
          max = value;
          max_idx = j;
        }
      }

      if (max_idx == testLabels[i]) {
        testCounter++;
      }
    }

    System.out.println("train accuracy " + ((double) trainCounter) / trainLabels.length);
    System.out.println("test accuracy " + ((double) testCounter) / testLabels.length);

    leNet.save(pathToModel);
  }

  private static void readMnist() {
    try {
      final Path fileTrain = Paths.get(LeNetModel.pathTrain);
      final Path fileTest = Paths.get(LeNetModel.pathTest);

      final int numTrainSamples = (int) Files.lines(fileTrain).count();
      trainLabels = new double[numTrainSamples];
      trainSamples = new VecBasedMx(numTrainSamples, heightIn * widthIn);

      final int numTestSamples = (int) Files.lines(fileTest).count();
      testLabels = new double[numTestSamples];
      testSamples = new VecBasedMx(numTestSamples, heightIn * widthIn);

      Files.lines(fileTrain)
          .forEach(new Consumer<String>() {
            int counter = 0;

            @Override
            public void accept(String s) {
              int[] data = Arrays.stream(s.split(","))
                .mapToInt(Integer::parseInt).toArray();
              trainLabels[counter] = data[0];
              for (int i = 1; i < data.length; i++) {
                trainSamples.set(counter, i - 1, data[i]);
              }
              counter++;
            }
          });

      Files.lines(fileTest)
          .forEach(new Consumer<String>() {
            int counter = 0;

            @Override
            public void accept(String s) {
              int[] data = Arrays.stream(s.split(","))
                  .mapToInt(Integer::parseInt).toArray();
              testLabels[counter] = data[0];
              for (int i = 1; i < data.length; i++) {
                testSamples.set(counter, i - 1, data[i]);
              }
              counter++;
            }
          });
    }
    catch (IOException e) {
      throw new RuntimeException(e);
    }
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

  private static void downloadFile(String url, String path) {
    try {
      final Path targetTrain = Paths.get(path);
      if (Files.notExists(targetTrain)) {
        URL website = new URL(url);
        try (InputStream in = website.openStream()) {
          Files.copy(in, targetTrain, StandardCopyOption.REPLACE_EXISTING);
        }
        catch (IOException e) {
          e.printStackTrace();
        }
      }
    } catch (MalformedURLException e) {
      e.printStackTrace();
    }
  }
}
