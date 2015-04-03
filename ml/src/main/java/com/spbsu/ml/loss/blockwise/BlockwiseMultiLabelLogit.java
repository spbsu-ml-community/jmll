package com.spbsu.ml.loss.blockwise;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.util.Combinatorics;
import com.spbsu.ml.BlockwiseFuncC1;
import com.spbsu.ml.TargetFunc;
import com.spbsu.ml.data.set.DataSet;

import static java.lang.Math.abs;
import static java.lang.Math.exp;

/**
 * User: qdeee
 * Date: 02.04.2015
 */
public class BlockwiseMultiLabelLogit extends BlockwiseFuncC1.Stub implements TargetFunc {
  private final Mx targets;

  public BlockwiseMultiLabelLogit(final Mx targets) {
    this.targets = targets;
  }

  @Override
  public void gradient(final Vec pointBlock, final Vec result, final int blockId) {
    double totalSum = 0.0;

    final Combinatorics.PartialPermutationsCheap enumerator = new Combinatorics.PartialPermutationsCheap(2, pointBlock.dim());
    enumerator.advance(); //skip all zeroes
    while (enumerator.advance()) {
      final int[] bits = enumerator.get();
      final double exp = exp(vecSumByMask(pointBlock, bits));
      totalSum += exp;
      for (int j = 0; j < pointBlock.dim(); j++) {
        if (bits[j] == 1) {
          result.adjust(j, exp);
        }
      }
    }

    VecTools.scale(result, 1. / (1. + totalSum));
    for (int j = 0; j < pointBlock.dim(); j++) {
      if (abs(targets.get(blockId, j) - 1) < 1e-10) {
        result.adjust(j, -1.);
      }
    }
  }

  @Override
  public double value(final Vec pointBlock, final int blockId) {
    double result = VecTools.multiply(targets.row(blockId), pointBlock);

    double sum = 0.0;
    final Combinatorics.PartialPermutationsCheap enumerator = new Combinatorics.PartialPermutationsCheap(2, pointBlock.dim());
    while (enumerator.advance()) {
      sum += exp(vecSumByMask(pointBlock, enumerator.get()));
    }
    result -= Math.log(1 + sum);

    return result;
  }

  private static double vecSumByMask(final Vec vec, final int[] mask) {
    double v = 0.;
    for (int i = 0; i < mask.length; i++) {
      if (mask[i] == 1) {
        v += vec.get(i);
      }
    }
    return v;
  }

  public Mx getTargets() {
    return targets;
  }

  @Override
  public double transformResultValue(final double value) {
    return exp(value / targets.length());
  }

  @Override
  public int blockSize() {
    return targets.columns();
  }

  @Override
  public int dim() {
    return targets.dim();
  }

  @Override
  public DataSet<?> owner() {
    return null;
  }
}
