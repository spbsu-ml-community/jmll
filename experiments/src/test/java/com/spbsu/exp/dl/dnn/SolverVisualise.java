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
import java.io.File;
import java.io.IOException;
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

  public static void main(String[] args) throws InterruptedException, IOException {
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

    final Solver solver = new NetBuilder().buildSolver(new File("experiments/src/test/resources/log_net.json"));
    solver.init();

    for (int i = 0; i < epochs; i++) {
      solver.solve(X, Y);

      final List<Double> result = new ArrayList<>(examples);
      for (int j = 0; j < examples; j++) {
        solver.net.input = new VecBasedMx(1, new ArrayVec(X.get(j, 0)));
        solver.net.forward();
        result.add(solver.net.output.get(0));
      }
      chartPanel.updateSeries(RESULT_SERIES, xSeries, result);
    }
  }

}
