package com.spbsu.exp.dl.dnn.rectifiers;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.xeiam.xchart.QuickChart;
import com.xeiam.xchart.SwingWrapper;

/**
 * jmll
 *
 * @author ksenon
 */
public class SigmoidVisualise {

  public static void main(final String[] args) {
    final Sigmoid sigmoid = new Sigmoid();

    final int r = 100;
    final int c = 100;
    final double min = -10;
    final double max =  10;
    final double step = (max - min) / (r * c);

    final Mx x = new VecBasedMx(r, c);
    for (int i = 0; i < r * c; i++) {
      x.set(i, min + (i * step));
    }
    final Mx y = new VecBasedMx(r, c);

    sigmoid.value(x, y);

    new SwingWrapper(QuickChart.getChart(
        "Sigmoid", "X", "Y", "σ(x)", x.toArray(), y.toArray()
    )).displayChart();
  }

}
