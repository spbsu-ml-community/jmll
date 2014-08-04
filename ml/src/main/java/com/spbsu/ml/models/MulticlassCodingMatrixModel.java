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
import com.spbsu.ml.func.FuncJoin;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import static java.lang.Math.exp;
import static java.lang.Math.log;

/**
 * User: qdeee
 * Date: 04.06.14
 * Time: 10:59
 */
public class MulticlassCodingMatrixModel extends MCModel.Stub {
  public static final Logger LOG = Logger.create(MulticlassCodingMatrixModel.class);
  protected final FuncJoin binaryClassifiers;
  protected final Mx codingMatrix;
  protected final double ignoreThreshold;
  protected final Metric<Vec> metric;

  public MulticlassCodingMatrixModel(final Mx codingMatrix, Func[] binaryClassifiers, double ignoreTreshold) {
    LOG.assertTrue(codingMatrix.columns() == binaryClassifiers.length, "Coding matrix columns count must match binary classifiers.");
    this.binaryClassifiers = new FuncJoin(binaryClassifiers);
    this.codingMatrix = codingMatrix;
    this.ignoreThreshold = ignoreTreshold;
    this.metric = new LossBasedSkipZeroMetric();
  }

  public FuncJoin getInternalModel() {
    return binaryClassifiers;
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
  public Vec probs(final Vec x) {
    throw new NotImplementedException();
  }

  @Override
  public int bestClass(final Vec x) {
    final Vec trans = binaryClassifiers.trans(x);
    final double[] dist = calcDistances(trans);
    return ArrayTools.min(dist);
  }

  @Override
  public int dim() {
    return binaryClassifiers.xdim();
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
