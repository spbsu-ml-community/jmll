package com.spbsu.ml.models;

import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.math.metrics.Metric;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecIterator;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.commons.util.logging.Logger;
import com.spbsu.ml.Func;

import static java.lang.Math.exp;
import static java.lang.Math.log;

/**
 * User: solar
 * Date: 10.01.14
 * Time: 10:59
 */
public class MulticlassCodingMatrixModel extends MultiClassModel {
  public static final Logger LOG = Logger.create(MulticlassCodingMatrixModel.class);
  protected final Mx codingMatrix;
  protected final double ignoreThreshold;
  private final Metric<Vec> metric;

  public MulticlassCodingMatrixModel(final Mx codingMatrix, Func[] binaryClassifiers, double ignore_treshold) {
    super(binaryClassifiers);
    LOG.assertTrue(codingMatrix.columns() == binaryClassifiers.length, "Coding matrix columns count must match binary classifiers.");
    this.codingMatrix = codingMatrix;
    this.ignoreThreshold = ignore_treshold;
    metric = new LossBasedSkipZeroMetric();
  }

  private Vec binarize(final Vec trans) {
    final Vec copy = VecTools.copy(trans);
    for (VecIterator it = copy.nonZeroes(); it.advance(); ) {
      if (Math.abs(it.value()) > ignoreThreshold)
        it.setValue(Math.signum(it.value()));
      else
        it.setValue(0.0);
    }
    return copy;
  }

  protected double[] calcDistances(Vec trans) {
    final double[] dist = new double[codingMatrix.rows()];
    final Vec binarize = binarize(trans);
    for (int i = 0; i < dist.length; i++) {
      dist[i] = metric.distance(binarize, codingMatrix.row(i));
    }
    return dist;
  }

  @Override
  public int bestClass(final Vec x) {
    final Vec trans = trans(x);
    final double[] dist = calcDistances(trans);
    return ArrayTools.min(dist);
  }

  protected static class LossBasedSkipZeroMetric implements Metric<Vec> {
    @Override
    public double distance(final Vec trans, final Vec row) {
      double result = 0;
      for (int l = 0; l < trans.dim(); l++) {
        final double prob = MathTools.sigmoid(trans.get(l));
        final double code = row.get(l);
        if (code > 0)
          result += log(prob);
        else if (code < 0)
          result += log(1 - prob);
      }
      return 1 - exp(result / trans.dim());
    }
  }
}
