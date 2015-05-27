package com.spbsu.exp.dl.dnn;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.exp.dl.dnn.rectifiers.BipolarSigmoid;
import com.spbsu.exp.dl.dnn.rectifiers.Flat;
import com.spbsu.exp.dl.dnn.rectifiers.Sigmoid;
import com.xeiam.xchart.Chart;
import com.xeiam.xchart.Series;
import com.xeiam.xchart.SeriesMarker;
import com.xeiam.xchart.XChartPanel;

import javax.swing.*;
import javax.swing.text.MaskFormatter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

/**
 * jmll
 *
 * @author ksenon
 */
public class SolverVisualise {

  public static final String TARGET_SERIES = "Target";
  public static final String RESULT_SERIES = "Result";

  public static void main(String[] args) throws InterruptedException {
    final Chart chart = new Chart(1000, 700);
    chart.setChartTitle("Sample Real-time Chart");
    chart.setXAxisTitle("X");
    chart.setYAxisTitle("Y");

    final Series targetSeries = chart.addSeries(TARGET_SERIES, new double[]{1}, new double[]{1});
    targetSeries.setMarker(SeriesMarker.NONE);

    final Series resultSeries = chart.addSeries(RESULT_SERIES, new double[]{1}, new double[]{1});
    resultSeries.setMarker(SeriesMarker.NONE);

    final XChartPanel chartPanel = new XChartPanel(chart);

    SwingUtilities.invokeLater(new Runnable() {
      @Override
      public void run() {
        JFrame frame = new JFrame("XChart");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.add(chartPanel);
        frame.pack();
        frame.setVisible(true);
      }
    });

    final int epochs = 100;
    final int examples = 1000;
    final double learningRate = 0.07;

    final Random random = new Random(1);
    final double[] data = new double[examples];
    for (int i = 0; i < examples; i++) {
      data[i] = random.nextDouble() * 10 + 1;
    }
    Arrays.sort(data);

    final Mx X = new VecBasedMx(examples, 1);
    final Mx Y = new VecBasedMx(examples, 1);
    for (int i = 0; i < examples; i++) {
      X.set(i, 0, data[i]);
      Y.set(i, 0, Math.log(data[i]));
    }
    final List<Double> xSeries = new ArrayList<>(examples);
    final List<Double> ySeries = new ArrayList<>(examples);
    for (int i = 0; i < examples; i++) {
      xSeries.add(data[i]);
      ySeries.add(Math.log(data[i]));
    }
    chartPanel.updateSeries(TARGET_SERIES, xSeries, ySeries);

    final FullyConnectedNet net = getNet();
    final Solver solver = new Solver();
    solver.batchSize = 100;
    solver.epochsNumber = 10;
    solver.learningRate = learningRate;
    solver.net = net;
    solver.debug = true;

    for (int i = 0; i < epochs; i++) {
      solver.solve(X, Y);

      final List<Double> result = new ArrayList<>(examples);
      for (int j = 0; j < examples; j++) {
        net.input = new VecBasedMx(1, new ArrayVec(X.get(j, 0)));
        net.forward();
        result.add(net.output.get(0));
      }
      chartPanel.updateSeries(RESULT_SERIES, xSeries, result);
    }
  }

  private static FullyConnectedNet getNet() {
    final Layer layer_1 = new Layer();
    layer_1.isTrain = true;
    layer_1.bias = 1;
    layer_1.bias_b = 1;
    layer_1.dropoutFraction = 0;
    layer_1.rectifier = new Sigmoid();
    layer_1.weights = init(200, 2);
    layer_1.difference = new VecBasedMx(200, 2);
    layer_1.activations = new VecBasedMx(100, 200);
    layer_1.input = new VecBasedMx(100, 1);
    layer_1.output = new VecBasedMx(100, 200);
    layer_1.debug = true;

    final Layer layer_2 = new Layer();
    layer_2.isTrain = true;
    layer_2.bias = 1;
    layer_2.bias_b = 0;
    layer_2.dropoutFraction = 0;
    layer_2.rectifier = new Sigmoid();
    layer_2.weights = init(50, 201);
    layer_2.difference = new VecBasedMx(50, 201);
    layer_2.activations = new VecBasedMx(100, 50);
    layer_2.input = new VecBasedMx(100, 200);
    layer_2.output = new VecBasedMx(100, 50);
    layer_2.debug = true;

    final Layer layer_3 = new Layer();
    layer_3.isTrain = true;
    layer_3.dropoutFraction = 0;
    layer_3.rectifier = new Flat();
    layer_3.weights = init(1, 50);
    layer_3.difference = new VecBasedMx(1, 50);
    layer_3.activations = new VecBasedMx(100, 1);
    layer_3.input = new VecBasedMx(100, 50);
    layer_3.output = new VecBasedMx(100, 1);
    layer_3.debug = true;

    final FullyConnectedNet net = new FullyConnectedNet();
    net.layers = new Layer[]{layer_1, layer_2, layer_3};
    net.input = new VecBasedMx(100, 1);
    net.output = new VecBasedMx(100, 1);
    net.debug = true;

    return net;
  }

  private static Mx init(final int r, final int c) {
    final VecBasedMx mx = new VecBasedMx(r, c);
    for (int i = 0; i < mx.dim(); i++) {
      mx.set(i, ThreadLocalRandom.current().nextDouble(0.001) - 0.0005);
    }
    return mx;
  }

}
