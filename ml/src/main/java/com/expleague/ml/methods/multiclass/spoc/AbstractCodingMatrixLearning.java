package com.expleague.ml.methods.multiclass.spoc;

import com.expleague.commons.math.vectors.Mx;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.data.tools.MCTools;
import com.expleague.ml.loss.blockwise.BlockwiseMLLLogit;

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

  public AbstractCodingMatrixLearning(final int k, final int l, final double lambdaC, final double lambdaR, final double lambda1) {
    this.k = k;
    this.l = l;
    this.lambdaC = lambdaC;
    this.lambdaR = lambdaR;
    this.lambda1 = lambda1;
  }

  public Mx trainCodingMatrix(final Mx similarityMatrix) {
    return findMatrixB(similarityMatrix);
  }

  public Mx trainCodingMatrix(final VecDataSet learn, final BlockwiseMLLLogit target) {
    final Mx similarityMatrix = MCTools.createSimilarityMatrix(learn, target.labels());
    return findMatrixB(similarityMatrix);
  }

  protected abstract Mx findMatrixB(Mx S);
}
