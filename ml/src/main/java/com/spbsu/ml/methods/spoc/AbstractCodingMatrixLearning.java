package com.spbsu.ml.methods.spoc;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.tools.MCTools;
import com.spbsu.ml.loss.blockwise.BlockwiseMLLLogit;

/**
 * User: qdeee
 * Date: 04.06.14
 */
public abstract class AbstractCodingMatrixLearning {
  protected final int k;
  protected final int l;
  protected final double lambdaC;
  protected final double lambdaR;
  protected final double lambda1;

  public AbstractCodingMatrixLearning(int k, int l, double lambdaC, double lambdaR, double lambda1) {
    this.k = k;
    this.l = l;
    this.lambdaC = lambdaC;
    this.lambdaR = lambdaR;
    this.lambda1 = lambda1;
  }

  public static boolean checkConstraints(final Mx B) {
    for (int l = 0; l < B.columns(); l++) {
      double sumPositive = 0;
      double sumNegative = 0;
      for (int k = 0; k < B.rows(); k++) {
        final double code = B.get(k, l);
        final double absCode = Math.abs(code);
        if (absCode > 1)
          return false;
        sumPositive += absCode + code;
        sumNegative += absCode - code;
      }
      if (sumPositive < 2 || sumNegative < 2)
        return false;
    }
    for (int k = 0; k < B.rows(); k++) {
      final double sum = VecTools.l1(B.row(k));
      if (sum < 1)
        return false;
    }
    return true;
  }

  public Mx trainCodingMatrix(final Mx similarityMatrix) {
    return findMatrixB(similarityMatrix);
  }

  public Mx trainCodingMatrix(final VecDataSet learn, BlockwiseMLLLogit target) {
    final Mx similarityMatrix = MCTools.createSimilarityMatrix(learn, target.labels());
    return findMatrixB(similarityMatrix);
  }

  protected abstract Mx findMatrixB(Mx S);
}
