package com.expleague.ml.models;

import com.expleague.commons.math.DiscontinuousTrans;
import com.expleague.commons.math.Trans;
import com.expleague.commons.math.vectors.SingleValueVec;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.random.FastRandom;
import com.expleague.ml.GridTools;
import com.expleague.ml.ProgressHandler;
import com.expleague.ml.data.set.impl.VecDataSetImpl;
import com.expleague.ml.func.Ensemble;
import com.expleague.ml.loss.L2;
import com.expleague.ml.loss.WeightedL2;
import com.expleague.ml.methods.BootstrapOptimization;
import com.expleague.ml.methods.GradientBoosting;
import com.expleague.ml.methods.trees.GreedyObliviousTree;
import com.expleague.ml.optimization.FuncConvex;
import com.expleague.ml.optimization.impl.GradientDescent;
import com.xeiam.xchart.Chart;
import com.xeiam.xchart.QuickChart;
import com.xeiam.xchart.SwingWrapper;
import org.junit.Test;

import java.util.function.Consumer;

import static org.junit.Assert.assertEquals;

public class DiffTreeTest {
  private static final int numSamples = 10_000;
  private static final double minX = 0;
  private static final double maxX = Math.PI;
  private static final double[] x = new double[numSamples];
  private static final double[] y = new double[numSamples];
  private static final double[] xVal = new double[numSamples];
  private static final double[] yVal = new double[numSamples];
  private static final FastRandom rng = new FastRandom();
  private static final double EPS = 0.05;
  private static final double step = (maxX - minX) / numSamples;

  static {
    for (int i = 0; i < numSamples; i++) {
      x[i] = minX + i * step;
      y[i] = Math.sin(x[i]);

      xVal[i] = x[i] + step / 2.;
      yVal[i] = Math.sin(xVal[i]);
    }
  }
  
  @Test
  public void synteticTest() {
    final VecDataSetImpl learn = new VecDataSetImpl(new VecBasedMx(1, new ArrayVec(x)), null);
    final VecDataSetImpl validate = new VecDataSetImpl(new VecBasedMx(1, new ArrayVec(xVal)), null);
    final Vec target = new ArrayVec(y);
    final Vec valTarget = new ArrayVec(yVal);

    final GradientBoosting<L2> boosting = new GradientBoosting<>(
        new BootstrapOptimization<>(new GreedyObliviousTree<>
            (GridTools.medianGrid(learn, 64), 6), rng),
        L2.class, 150, 0.05);

    final Consumer<Trans> counter = new ProgressHandler() {
      int index = 0;

      @Override
      public void accept(Trans partial) {
        final Vec result = partial.transAll(learn.data()).vec();
        final Vec resultVal = partial.transAll(validate.data()).vec();
        final double mse = VecTools.distanceL2(result, target);
        final double mseVal = VecTools.distanceL2(resultVal, valTarget);
        System.out.println("[" + (index++) + "] learn: " + mse + " val: " + mseVal);
      }
    };
    boosting.addListener(counter);
    final Ensemble ensemble = boosting.fit(learn, new WeightedL2(target, learn));
    final DiscontinuousTrans subgradient = ensemble.subgradient();
    assert subgradient != null;

    final double[] result = new double[x.length];
    for (int i = 0; i < x.length; i++) {
      result[i] = ensemble.trans(new SingleValueVec(x[i])).get(0);
    }

    Chart chart = QuickChart.getChart("Sample Chart", "X", "Y", "y(x)", x, result);
    new SwingWrapper(chart).displayChart();

    final GradientDescent optimizer = new GradientDescent(new ArrayVec(0.1), 1e-3);
    double xo = optimizer.optimize(new FuncConvex.Stub() {
      @Override
      public double getGradLipParam() {
        return 100;
      }

      @Override
      public Vec gradientTo(Vec x_, Vec to) {
        // FIXME: only left derivative?
        final double gridStep = Math.PI / 64.;
        final double left = subgradient.left(x_).get(0);
        final double right = subgradient.right(x_).get(0);
        final double x = x_.get(0);
        final double leftGrad = (Math.sin(x) - Math.sin(x - gridStep));
        final double rightGrad = (Math.sin(x + gridStep) - Math.sin(x));
        to.set(0, -left);

        assertEquals(leftGrad, left, EPS);
        assertEquals(rightGrad, right, EPS);
        return to;
      }

      @Override
      public double value(Vec x) {
        return ensemble.trans(x).get(0);
      }

      @Override
      public int dim() {
        return ensemble.xdim();
      }
    }).get(0);

    assertEquals(Math.PI / 2, xo, EPS);
  }
}
