package com.spbsu.ml.cuda;

import com.xeiam.xchart.*;
import jcuda.jcurand.curandGenerator;

import java.util.ArrayList;
import java.util.Collection;

/**
 * Project jmll
 *
 * @author Ksen
 */
public class Charts {

  public static void main(final String[] args) {
    final curandGenerator generator = JCurandHelper.createDefault();
    final float[] floats = JCurandHelper.generateUniformHost(1_000_000, generator);
    JCurandHelper.destroyGenerator(generator);

    final Chart chart = new ChartBuilder()
        .chartType(StyleManager.ChartType.Bar)
        .width(800).height(600)
        .title("Default Uniform")
        .xAxisTitle("value")
        .yAxisTitle("hits")
        .build()
    ;
    final Histogram histogram = new Histogram(transform(floats), 100);
    chart.addSeries("Test", histogram.getxAxisData(), histogram.getyAxisData());
    new SwingWrapper(chart).displayChart();
  }

  private static Collection<Double> transform(final float[] array) {
    final ArrayList<Double> dArray = new ArrayList<>(array.length);
    for (int i = 0; i < array.length; i++) {
      dArray.add((double)array[i]);
    }
    return dArray;
  }

}
