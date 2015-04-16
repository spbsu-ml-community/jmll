package com.spbsu.exp.dl;

import com.spbsu.ml.cuda.root.JCudaVectorInscale;
import com.xeiam.xchart.BitmapEncoder;
import com.xeiam.xchart.Chart;
import com.xeiam.xchart.QuickChart;

import java.io.IOException;
import java.text.MessageFormat;

/**
 * jmll
 * ksenon, 18:50 16.04.2015
 */
public class Invoker {

  public static void main(final String[] args) throws IOException { // very very very rough
    if (args.length != 2) {
      throw new IllegalArgumentException("<MAX_ARRAY_LENGTH> <WHERE_TO_SAVE_CHART>");
    }
    final int maxSize = Integer.parseInt(args[0]) * 4;
    final String chartPath = args[1];

    JCudaVectorInscale.exp(new double[]{1, 2, 3});

    final int increment = 1024;
    final double[] X = new double[(maxSize / increment) + 1];
    final double[][] Y = new double[2][(maxSize / increment) + 1];
    for (int i = 4, counter = 0; i < maxSize; i += increment, counter++) {
      final double[] cpuArray = get(i);
      final double[] gpuArray = new double[cpuArray.length];
      System.arraycopy(cpuArray, 0, gpuArray, 0, cpuArray.length);

      final long cpuBegin = System.nanoTime();
      exp(cpuArray);
      final long cpuEnd = System.nanoTime() - 20;

      final long gpuBegin = System.nanoTime();
      JCudaVectorInscale.exp(gpuArray);
      final long gpuEnd = System.nanoTime() - 20;

      final long cpuTime = cpuEnd - cpuBegin;
      final long gpuTime = gpuEnd - gpuBegin;
      System.out.println(MessageFormat.format(
          "Size = {0}, CentrPU = {1}, GraphPU = {2}, Delta = {3}", i, cpuTime, gpuTime, compare(cpuArray, gpuArray)
      ));
      Y[0][counter] = cpuTime;
      Y[1][counter] = gpuTime;
      X[counter] = i;
    }
    final Chart chart = QuickChart.getChart(
        "CPU vs GPU", "Length of Array", "nano", new String[]{"CentrPU", "GraphPU"}, X, Y
    );
    BitmapEncoder.saveBitmap(chart, chartPath, BitmapEncoder.BitmapFormat.PNG);
  }

  private static double[] get(final int size) {
    final double[] array = new double[size];
    for (int i = 0; i < size; i++) {
      array[i] = Math.random();
    }
    return array;
  }

  private static void exp(final double[] array) {
    for (int i = 0; i < array.length; i += 4) {
      array[i] = Math.exp(array[i]);
      array[i + 1] = Math.exp(array[i + 1]);
      array[i + 2] = Math.exp(array[i + 2]);
      array[i + 3] = Math.exp(array[i + 3]);
    }
  }

  private static double compare(final double[] a, final double[] b) { //broken associativity, but who cares
    double delta1 = 0;
    double delta2 = 0;
    double delta3 = 0;
    double delta4 = 0;
    for (int i = 0; i < a.length; i += 4) {
      delta1 += a[i] - b[i];
      delta2 += a[i + 1] - b[i + 1];
      delta3 += a[i + 2] - b[i + 2];
      delta4 += a[i + 3] - b[i + 3];
    }
    return (delta1 + delta2) + (delta3 + delta4);
  }

}
