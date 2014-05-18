package com.spbsu.ml.models;

import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.commons.util.Pair;
import com.spbsu.commons.util.logging.Logger;
import com.spbsu.ml.Func;

/**
 * User: solar
 * Date: 10.01.14
 * Time: 10:59
 */
public class MulticlassCodingMatrixModel extends MultiClassModel {
  public static final Logger LOG = Logger.create(MulticlassCodingMatrixModel.class);
  private final Mx codingMatrix;
//  private final int dim;

  public MulticlassCodingMatrixModel(final Mx codingMatrix, Func[] binaryClassifiers) {
    super(binaryClassifiers);
    LOG.assertTrue(codingMatrix.columns() == binaryClassifiers.length, "Coding matrix columns count must match binary classifiers.");
    this.codingMatrix = codingMatrix;
  }

  public Pair<double[], int[]> calcDistance(Vec trans) {
    final double[] dist = new double[codingMatrix.rows()];
    for (int i = 0; i < dist.length; i++) {
      dist[i] = VecTools.distance(trans, codingMatrix.row(i));
    }
    final int[] idxs = ArrayTools.sequence(0, dist.length);
    ArrayTools.parallelSort(dist, idxs);
    return Pair.create(dist, idxs);
  }

  @Override
  public int bestClass(final Vec x) {
    Vec trans = trans(x);
    final Pair<double[], int[]> dist = calcDistance(trans);
    return dist.second[0];
  }
}
