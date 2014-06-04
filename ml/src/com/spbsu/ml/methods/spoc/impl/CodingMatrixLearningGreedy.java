package com.spbsu.ml.methods.spoc.impl;

import com.spbsu.commons.math.vectors.*;
import com.spbsu.commons.math.vectors.impl.VecBasedMx;
import com.spbsu.commons.util.Combinatorics;
import com.spbsu.ml.methods.spoc.AbstractCodingMatrixLearning;
import sun.plugin.dom.exception.InvalidStateException;

/**
 * User: qdeee
 * Date: 07.05.14
 */
public class CodingMatrixLearningGreedy extends AbstractCodingMatrixLearning {
  public CodingMatrixLearningGreedy(int k, int l, double lambdaC, double lambdaR, double lambda1) {
    super(k, l, lambdaC, lambdaR, lambda1);
  }

  private double calcLoss(final Mx B, final Mx S) {
    double result = 0;
    final Mx mult = VecTools.multiply(B, VecTools.transpose(B));
    result -= VecTools.trace(VecTools.multiply(mult, S));
    result += lambdaR * VecTools.sum(mult);
    result += lambdaC * VecTools.sum2(B);
    result += lambda1 * VecTools.l1(B);
    return result;
  }

  @Override
  public Mx findMatrixB(final Mx S) {
    final Mx mxB = new VecBasedMx(k, l);
    for (int j = 0; j < l; j++) {
      final Combinatorics.PartialPermutations permutationsGenerator = new Combinatorics.PartialPermutations(3, k);
      int[] bestPerm = null;
      double bestLoss = Double.MAX_VALUE;
      while (permutationsGenerator.hasNext()) {
        final int[] perm = permutationsGenerator.next();
        for (int i = 0; i < k; i++) {
          mxB.set(i, j, perm[i] - 1);
        }
        final Mx sub = mxB.sub(0, 0, k, j + 1);
        if (checkConstraints(sub) && checkColumnsIndependence(sub)) {
          final double loss = calcLoss(sub, S);
          if (loss < bestLoss) {
            bestLoss = loss;
            bestPerm = perm;
          }
        }
      }
      if (bestPerm != null) {
        for (int i = 0; i < k; i++) {
          mxB.set(i, j, bestPerm[i] - 1);
        }
      }
      else
        throw new InvalidStateException("Not found appreciate column #" + j);
      System.out.println("Column " + j + " is over!");
    }
    return mxB;
  }

  //check pairwise columns independence
  public static boolean checkColumnsIndependence(final Mx B) {
    for (int j1 = 0; j1 < B.columns(); j1++) {
      final Vec col1 = B.col(j1);
      final double norm1 = VecTools.norm(col1);
      if (norm1 == 0)
        return false;
      for (int j2 = j1 + 1; j2 < B.columns(); j2++) {
        final Vec col2 = B.col(j2);
        final double norm2 = VecTools.norm(col2);
        if (norm2 == 0)
          return false;
        final double cosine = VecTools.multiply(col1, col2) / (norm1 * norm2);
        if (Math.abs(cosine) > 0.999)
          return false;
      }
    }
    return true;
  }
}