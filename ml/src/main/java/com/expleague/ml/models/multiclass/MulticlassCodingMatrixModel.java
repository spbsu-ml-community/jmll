package com.expleague.ml.models.multiclass;

import com.expleague.commons.math.MathTools;
import com.expleague.commons.math.metrics.Metric;
import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecIterator;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.util.ArrayTools;
import com.expleague.commons.math.Func;
import com.expleague.ml.func.FuncJoin;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static java.lang.Math.exp;
import static java.lang.Math.log;

/**
 * User: qdeee
 * Date: 04.06.14
 * Time: 10:59
 */
public class MulticlassCodingMatrixModel extends MCModel.Stub {
  public static final Logger LOG = LoggerFactory.getLogger(MulticlassCodingMatrixModel.class);
  protected final FuncJoin binaryClassifiers;
  protected final Mx codingMatrix;
  protected final double ignoreThreshold;
  protected final Metric<Vec> metric;

  public MulticlassCodingMatrixModel(final Mx codingMatrix, final Func[] binaryClassifiers, final double ignoreTreshold) {
    if (codingMatrix.columns() != binaryClassifiers.length)
      throw new IllegalArgumentException("Coding matrix columns count must match binary classifiers.");
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
    for (final VecIterator it = copy.nonZeroes(); it.advance(); ) {
      if (Math.abs(it.value()) > ignoreThreshold)
        it.setValue(Math.signum(it.value()));
      else
        it.setValue(0.0);
    }
    return copy;
  }

  protected double[] calcDistances(final Vec trans) {
    final double[] dist = new double[codingMatrix.rows()];
    final Vec binarize = binarize(trans);
    for (int i = 0; i < dist.length; i++) {
      dist[i] = metric.distance(binarize, codingMatrix.row(i));
    }
    return dist;
  }

  @Override
  public int countClasses() {
    return codingMatrix.rows();
  }

  @Override
  public Vec probs(final Vec x) {
    final Vec trans = binaryClassifiers.trans(x);
    final double[] dist = calcDistances(trans);
    for (int i = 0; i < dist.length; i++) {
      dist[i] = 1 - dist[i];
    }
    return new ArrayVec(dist);
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

  public Mx getCodingMatrix() {
    return codingMatrix;
  }
}
