package com.expleague.dl4j;

import com.google.common.graph.NetworkBuilder;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Arrays;

public class Bench {
  private static int NUM_SHOTS = 200;
  public static void main(String[] args) {
    int nOut = 1024;
    xxx(1024);
    xxx(2048);
    xxx(4096);

    NetworkBuilder builder = null;
  }

  private static void xxx(int nOut) {
    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(1)
        .updater(new Sgd(1e-3)).l1(0.3).l2(1e-3).list()
        .layer(0, new org.deeplearning4j.nn.conf.layers.DenseLayer.Builder().nIn(1).nOut(nOut)
            .activation(Activation.IDENTITY).weightInit(WeightInit.XAVIER).build())
        .layer(1, new org.deeplearning4j.nn.conf.layers.DenseLayer.Builder().nIn(nOut).nOut(nOut)
            .activation(Activation.IDENTITY).weightInit(WeightInit.XAVIER).build())
        .layer(2, new org.deeplearning4j.nn.conf.layers.DenseLayer.Builder().nIn(nOut).nOut(nOut)
            .activation(Activation.IDENTITY).weightInit(WeightInit.XAVIER).build())
        .layer(3, new org.deeplearning4j.nn.conf.layers.DenseLayer.Builder().nIn(nOut).nOut(nOut)
            .activation(Activation.IDENTITY).weightInit(WeightInit.XAVIER).build())
        .layer(4, new org.deeplearning4j.nn.conf.layers.DenseLayer.Builder().nIn(nOut).nOut(nOut)
            .activation(Activation.IDENTITY).weightInit(WeightInit.XAVIER).build())
//        .layer(5, new org.deeplearning4j.nn.conf.layers.DenseLayer.Builder().nIn(4096).nOut(4096)
//            .activation(Activation.IDENTITY).weightInit(WeightInit.XAVIER).build())
        .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
            .weightInit(WeightInit.XAVIER).nIn(nOut).nOut(1).build())
        .build();

    MultiLayerNetwork model = new MultiLayerNetwork(conf);
    model.init();
    INDArray array = Nd4j.linspace(1, 1, 1);

    double[] times = new double[NUM_SHOTS];
    for (int i = 0; i < NUM_SHOTS; i++) {
      final long start = System.nanoTime();
      model.output(array, false);
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
}
