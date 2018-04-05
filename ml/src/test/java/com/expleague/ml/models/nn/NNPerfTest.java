package com.expleague.ml.models.nn;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.ml.models.nn.layers.*;
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
    convPerfTest(5);
    convPerfTest(10);
    convPerfTest(20);
  }

  public void convPerfTest(int channels) {
    final int height = 70;
    final int width = 70;
    final NetworkBuilder<Vec>.Network network = new NetworkBuilder<>(
        new ConstSizeInput3D(height, width, channels))
        .append(ConvLayerBuilder.create().ksize(30, 30).channels(channels))
        .append(ConvLayerBuilder.create().ksize(15, 15).channels(channels))
        .append(ConvLayerBuilder.create().ksize(7, 7).channels(channels))
        .append(ConvLayerBuilder.create().ksize(5, 5).channels(channels))
        .append(ConvLayerBuilder.create().ksize(3, 3).channels(channels))
        .append(ConvLayerBuilder.create().ksize(1, 1).channels(channels))
        .append(ConvLayerBuilder.create().ksize(1, 1).channels(1))
        .build(new OneOutLayer());

    Vec input = new ArrayVec(height * width * channels);
    VecTools.fill(input, 1.);

    perfNNCheck(input, new ConvNet(network));
  }

  @Test
  public void perceptronMultiTest() {
    System.out.println("perceptron test, forward pass");
    perceptronPerfTest(1024);
    perceptronPerfTest(2048);
    perceptronPerfTest(4096);
  }

  public void perceptronPerfTest(int n_hid) {
    System.out.println("config: [" + n_hid + "]");

    NetworkBuilder<Vec>.Network network = new NetworkBuilder<>(new ConstSizeInput(1))
        .append(FCLayerBuilder.create().nOut(n_hid))
        .append(FCLayerBuilder.create().nOut(n_hid))
        .append(FCLayerBuilder.create().nOut(n_hid))
        .append(FCLayerBuilder.create().nOut(n_hid))
        .append(FCLayerBuilder.create().nOut(n_hid))
        .append(FCLayerBuilder.create().nOut(1))
        .build(new OneOutLayer());

    perfNNCheck(new ArrayVec(1.), new ConvNet(network));
  }

  private void perfNNCheck(Vec input, ConvNet nn) {
    double[] times = new double[NUM_SHOTS];
    for (int i = 0; i < NUM_SHOTS; i++) {
      final long start = System.nanoTime();
      input = nn.apply(input);
      final long finish = System.nanoTime();
      times[i] = (finish - start) / 1_000_000.;
    }

    System.out.println(stat(times));
  }

  private static Stat stat(double[] array) {
    Arrays.sort(array);

    int median = (array.length + 1) / 2;

    return new Stat(array[median], array[median / 2], array[median + median / 2]);
  }

}
