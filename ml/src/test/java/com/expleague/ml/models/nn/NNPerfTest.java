package com.expleague.ml.models.nn;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.ml.models.nn.layers.ConstSizeInput;
import com.expleague.ml.models.nn.layers.FCLayerBuilder;
import com.expleague.ml.models.nn.layers.OneOutLayer;
import org.junit.Test;

import java.util.Arrays;

public class NNPerfTest {
  private static final int NUM_SHOTS = 200;
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
  public void perceptronMultiTest() {
    System.out.println("perceptron test, forward pass");
    perceptronPerfTest(1024);
    perceptronPerfTest(2048);
    perceptronPerfTest(4096);
  }

  public void perceptronPerfTest(int n_hid) {
    System.out.println("config: [" + n_hid + "]");

    double[] times = new double[NUM_SHOTS];

    NetworkBuilder<Vec>.Network network = new NetworkBuilder<>(new ConstSizeInput(1))
        .append(FCLayerBuilder.create().nOut(n_hid))
        .append(FCLayerBuilder.create().nOut(n_hid))
        .append(FCLayerBuilder.create().nOut(n_hid))
        .append(FCLayerBuilder.create().nOut(n_hid))
        .append(FCLayerBuilder.create().nOut(n_hid))
        .append(FCLayerBuilder.create().nOut(1))
        .build(new OneOutLayer());

    final ConvNet nn = new ConvNet(network);
    Vec input = new ArrayVec(1.);

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
