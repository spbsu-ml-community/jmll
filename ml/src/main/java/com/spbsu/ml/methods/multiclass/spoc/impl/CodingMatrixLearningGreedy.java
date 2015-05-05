package com.spbsu.ml.methods.multiclass.spoc.impl;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.MxTools;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.util.Combinatorics;
import com.spbsu.ml.methods.multiclass.spoc.AbstractCodingMatrixLearning;
import com.spbsu.ml.methods.multiclass.spoc.CMLHelper;

/**
 * User: qdeee
 * Date: 07.05.14
 */
public class CodingMatrixLearningGreedy extends AbstractCodingMatrixLearning {
  public CodingMatrixLearningGreedy(final int k, final int l, final double lambdaC, final double lambdaR, final double lambda1) {
    super(k, l, lambdaC, lambdaR, lambda1);
  }

  protected double calcLoss(final Mx B, final Mx S) {
    double result = 0;
    final Mx mult = MxTools.multiply(B, MxTools.transpose(B));
    result -= MxTools.trace(MxTools.multiply(mult, S));
    result += lambdaR * VecTools.sum(mult);
    result += lambdaC * VecTools.sum2(B);
    result += lambda1 * VecTools.l1(B);
    return result;
  }

  @Override
  public Mx findMatrixB(final Mx S) {
    final Mx mxB = new VecBasedMx(k, l);
    for (int j = 0; j < l; j++) {
      final Combinatorics.PartialPermutations permutationsGenerator = new Combinatorics.PartialPermutations(2, k);
      int[] bestPerm = null;
      double bestLoss = Double.MAX_VALUE;
      while (permutationsGenerator.hasNext()) {
        final int[] perm = permutationsGenerator.next();
        for (int i = 0; i < k; i++) {
          mxB.set(i, j, 2 * perm[i] - 1);
        }
        final Mx sub = mxB.sub(0, 0, k, j + 1);
        if (CMLHelper.checkConstraints(sub) && CMLHelper.checkColumnsIndependence(sub)) {
          final double loss = calcLoss(sub, S);
          if (loss < bestLoss) {
            bestLoss = loss;
            bestPerm = perm;
          }
        }
      }
      if (bestPerm != null) {
        for (int i = 0; i < k; i++) {
          mxB.set(i, j, 2 * bestPerm[i] - 1);
        }
      }
      else
        throw new IllegalStateException("Not found appreciate column #" + j);
//      System.out.println("Column " + j + " is over!");
    }
    return mxB;
  }
}