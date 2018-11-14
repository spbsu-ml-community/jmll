package com.expleague.ml.models.nn;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.ml.func.generic.ReLU;
import com.expleague.ml.models.nn.layers.*;
import org.jetbrains.annotations.NotNull;
import org.junit.Test;

import java.util.Arrays;

public class NNPerfTest {
  private static final int NUM_SHOTS = 100;
  private static class Stat {
    public final double median;
    public final double quart1;
    public final double quart3;

    public Stat(double median, double quart1, double quart3) {
      this.median = median;
      this.quart1 = quart1;
      this.quart3 = quart3;
    }

    @Override
    public String toString() {
      return "median " + median + "; quartiles [" + quart1 + ", " + quart3 + "] ms";
    }
  }

  @Test
  public void convTest() {
    System.out.println("conv test, forward pass");
    perfNNCheck(createConvNet(5));
    perfNNCheck(createConvNet(10));
    perfNNCheck(createConvNet(20));
  }

  @Test
  public void alexNetTest() {
    System.out.println("alexnet test, forward pass");
    perfNNCheck(createAlexNet());
  }

  @Test
  public void perceptronMultiTest() {
    System.out.println("perceptron test, forward pass");
    perfNNCheck(createPerceptron(1024));
    perfNNCheck(createPerceptron(2048));
    perfNNCheck(createPerceptron(4096));
  }

  @Test
  public void convGradTest() {
    System.out.println("conv test, forward pass");
    perfNNGradCheck(createConvNet(5));
    perfNNGradCheck(createConvNet(10));
    perfNNGradCheck(createConvNet(20));
  }

  @Test
  public void alexNetGradTest() {
    System.out.println("alexnet test, forward pass");
    perfNNGradCheck(createAlexNet());
  }

  @Test
  public void perceptronMultiGradTest() {
    System.out.println("perceptron test, forward pass");
    perfNNGradCheck(createPerceptron(1024));
    perfNNGradCheck(createPerceptron(2048));
    perfNNGradCheck(createPerceptron(4096));
  }

  private static ConvNet createAlexNet() {
    NetworkBuilder<Vec>.Network network = new NetworkBuilder<>(
        new ConstSizeInput3D(224, 224, 3))
        .addSeq(
            ConvLayerBuilder.create()
                .channels(64).ksize(11, 11).stride(4, 4).padd(2, 2).activation(ReLU.class),
            PoolLayerBuilder.create()
                .ksize(3, 3).stride(2, 2),
            ConvLayerBuilder.create()
                .channels(192).ksize(5, 5).padd(2, 2).activation(ReLU.class),
            PoolLayerBuilder.create()
                .ksize(3, 3).stride(2, 2),
            ConvLayerBuilder.create()
                .channels(384).ksize(3, 3).padd(2, 2).activation(ReLU.class),
            ConvLayerBuilder.create()
                .channels(256).ksize(3, 3).activation(ReLU.class),
            PoolLayerBuilder.create()
                .ksize(3, 3).stride(2, 2),
            FCLayerBuilder.create()
                .nOut(4096).activation(ReLU.class),
            FCLayerBuilder.create()
                .nOut(4096).activation(ReLU.class),
            FCLayerBuilder.create()
                .nOut(1000)
        )
        .build(new OneOutLayer());

    final Vec weights = new ArrayVec(network.wdim());
    network.initWeights(weights);

    return new ConvNet(network, weights);
  }

  private static ConvNet createConvNet(int channels) {
    final NetworkBuilder<Vec>.Network network = new NetworkBuilder<>(
        new ConstSizeInput3D(70, 70, channels))
        .append(ConvLayerBuilder.create().ksize(30, 30).channels(channels))
        .append(ConvLayerBuilder.create().ksize(15, 15).channels(channels))
        .append(ConvLayerBuilder.create().ksize(7, 7).channels(channels))
        .append(ConvLayerBuilder.create().ksize(5, 5).channels(channels))
        .append(ConvLayerBuilder.create().ksize(3, 3).channels(channels))
        .append(ConvLayerBuilder.create().ksize(1, 1).channels(channels))
        .append(ConvLayerBuilder.create().ksize(1, 1).channels(1))
        .build(new OneOutLayer());

    final Vec weights = new ArrayVec(network.wdim());
    network.initWeights(weights);

    return new ConvNet(network, weights);
  }

  private static ConvNet createPerceptron(int n_hid) {
    NetworkBuilder<Vec>.Network network = new NetworkBuilder<>(new ConstSizeInput(1))
        .append(FCLayerBuilder.create().nOut(n_hid))
        .append(FCLayerBuilder.create().nOut(n_hid))
        .append(FCLayerBuilder.create().nOut(n_hid))
        .append(FCLayerBuilder.create().nOut(n_hid))
        .append(FCLayerBuilder.create().nOut(n_hid))
        .append(FCLayerBuilder.create().nOut(1))
        .build(new OneOutLayer());
    final Vec weights = new ArrayVec(network.wdim());
    network.initWeights(weights);
    return new ConvNet(network, weights);
  }

  private void perfNNCheck(ConvNet nn) {
    System.out.println(nn);
    Vec input = new ArrayVec(nn.xdim());
    VecTools.fill(input, 1.);

    double[] times = new double[NUM_SHOTS];
    for (int i = 0; i < NUM_SHOTS; i++) {
      final long start = System.nanoTime();
      input = nn.apply(input);
      final long finish = System.nanoTime();
      times[i] = (finish - start) / 1_000_000.;
    }

    System.out.println(stat(times));
  }

  private void perfNNGradCheck(ConvNet nn) {
    System.out.println(nn);
    Vec input = new ArrayVec(nn.xdim());
    VecTools.fill(input, 1.);

    double[] times = new double[NUM_SHOTS];
    Vec grad = new ArrayVec(nn.wdim());

    for (int i = 0; i < NUM_SHOTS; i++) {
      final long start = System.nanoTime();
      grad = nn.gradientTo(input, grad);
      final long finish = System.nanoTime();
      times[i] = (finish - start) / 1_000_000.;
    }

    System.out.println(stat(times));
  }

  @NotNull
  private static Stat stat(double[] array) {
    Arrays.sort(array);
    int median = (array.length + 1) / 2;
    return new Stat(array[median], array[median / 2], array[median + median / 2]);
  }

}
