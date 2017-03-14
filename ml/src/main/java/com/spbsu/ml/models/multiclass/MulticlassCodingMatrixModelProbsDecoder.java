package com.spbsu.ml.models.multiclass;

import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.Func;

/**
 * User: qdeee
 * Date: 23.05.14
 */
@Deprecated
public class MulticlassCodingMatrixModelProbsDecoder extends MulticlassCodingMatrixModel {
  private final Mx mu;

  public MulticlassCodingMatrixModelProbsDecoder(final Mx codingMatrix, final Func[] binaryClassifiers, final double ignoreTreshold, final Mx mu) {
    super(codingMatrix, binaryClassifiers, ignoreTreshold);
    this.mu = mu;
  }

  @Override
  protected double[] calcDistances(final Vec trans) {
    final double[] dist = new double[codingMatrix.rows()];
    for (int k = 0; k < dist.length; k++) {
      double value = 1.0;
      for (int l = 0; l < codingMatrix.columns(); l++) {
        final double sigmoid = MathTools.sigmoid(trans.get(l));
        value *= mu.get(k, l) * sigmoid + (1 - mu.get(k, l)) * (1 - sigmoid);
      }
      dist[k] = 1 - value;
    }
    return dist;
  }
}
