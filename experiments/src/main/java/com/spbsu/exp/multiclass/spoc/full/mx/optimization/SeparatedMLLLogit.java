package com.spbsu.exp.multiclass.spoc.full.mx.optimization;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecIterator;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.seq.IntSeq;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.commons.math.FuncC1;
import com.spbsu.ml.TargetFunc;
import com.spbsu.ml.data.set.DataSet;

import static java.lang.Math.exp;
import static java.lang.Math.log;

/**
 * User: qdeee
 * Date: 18.11.14
 */
public class SeparatedMLLLogit extends FuncC1.Stub implements TargetFunc {
  private final IntSeq target;
  private final DataSet<?> owner;
  private final int K;
  private final int L;

  public SeparatedMLLLogit(final int l, final IntSeq target, final DataSet<?> owner) {
    this.target = target;
    this.owner = owner;
    this.K = ArrayTools.max(target) + 1;
    this.L = l;
  }

  public SeparatedMLLLogit(final int l, final Vec target, final DataSet<?> owner) {
    final int[] intTarget = new int[target.length()];
    final VecIterator iter = target.nonZeroes();
    while (iter.advance()) {
      intTarget[iter.index()] = (int) iter.value();
    }
    this.target = new IntSeq(intTarget);
    this.owner = owner;
    this.K = ArrayTools.max(this.target) + 1;
    this.L = l;
  }

  public IntSeq labels() {
    return target;
  }

  public int getClassesCount() {
    return K;
  }

  public int getBinClassifiersCount() {
    return L;
  }

  @Override
  public DataSet<?> owner() {
    return owner;
  }

  /**
   *
   * @param x: x - block matrix, KxL - mxB part
   *                             NxL - f_i parts
   * @return separated mlllogit value
   */
  @Override
  public double value(final Vec x) {
    final Mx mxX = new VecBasedMx(L, x);
    final Mx mxB = mxX.sub(0, 0, K - 1, L);
    final Mx mxF = mxX.sub(K - 1, 0, target.length(), L);

    double result = 0.0;
    for (int i = 0; i < target.length(); i++) {
      final Vec fRow = mxF.row(i);
      double sum = 0.0;
      for (int c = 0; c < K - 1; c++) {
        sum += exp(VecTools.multiply(mxB.row(c), fRow));
      }

      final int pointTrueClass = target.intAt(i);
      if (pointTrueClass < K - 1) {
        final double classifiersValue = VecTools.multiply(mxB.row(pointTrueClass), fRow);
        result += log(exp(classifiersValue) / (1. + sum));
      } else {
        result += log(1. / (1. + sum));
      }
    }
    return result;
  }

  @Override
  public int dim() {
    return (K - 1 + target.length()) * L;
  }

  private double calcSum(final Mx B, final Vec fRow) {
    double sum = 1.0;
    for (int c = 0; c < K - 1; c++) {
      sum += exp(VecTools.multiply(B.row(c), fRow));
    }
    return sum;
  }

  @Override
  public Mx gradient(final Vec x) {
    final Mx grad = new VecBasedMx(K - 1 + target.length(), L);
    final Mx gradB = grad.sub(0, 0, K - 1, L);
    final Mx gradF = grad.sub(K - 1, 0, target.length(), L);

    final Mx mxX = new VecBasedMx(L, x);
    final Mx mxB = mxX.sub(0, 0, K - 1, L);
    final Mx mxF = mxX.sub(K - 1, 0, target.length(), L);

    for (int c = 0; c < K - 1; c++) {
      for (int j = 0; j < L; j++) {
        double gradElemValue = 0.0;
        for (int i = 0; i < target.length(); i++) {
          final Vec rowF = mxF.row(i);
          double value = -exp(VecTools.multiply(mxB.row(c), rowF)) / calcSum(mxB, rowF);
          if (target.intAt(i) == c) {
            value += 1;
          }
          gradElemValue += value * rowF.get(j);
        }
        gradB.set(c, j, gradElemValue);
      }
    }

    for (int i = 0; i < target.length(); i++) {
      final Vec rowF = mxF.row(i);
      for (int j = 0; j < L; j++) {
        double gradElemValue = mxB.get(target.intAt(i), j);
        double weightedSumExp = 0.0;
        for (int c = 0; c < K - 1; c++) {
          weightedSumExp += mxB.get(c, j) * exp(VecTools.multiply(mxB.row(c), rowF));
        }
        gradElemValue += weightedSumExp / calcSum(mxB, rowF);
        gradF.set(i, j, gradElemValue);
      }
    }

    return grad;
  }
}
