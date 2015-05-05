package com.spbsu.ml.methods.multiclass.spoc.impl;

import com.spbsu.commons.math.vectors.*;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.ml.methods.multiclass.spoc.AbstractCodingMatrixLearning;
import com.spbsu.ml.methods.multiclass.spoc.CMLHelper;

import java.util.Random;

/**
 * User: qdeee
 * Date: 07.05.14
 */
public class CodingMatrixLearning extends AbstractCodingMatrixLearning {
  private static final double MX_IGNORE_THRESHOLD = 0.1;
  private static final double MX_LEARN_EPS = 1e-3;


  private final Mx initB;

  private final double mxLearnStep;


  public CodingMatrixLearning(final Mx initB, final double mxLearnStep, final double lambdaC, final double lambdaR, final double lambda1) {
    super(initB.rows(), initB.columns(), lambdaC, lambdaR, lambda1);
    this.initB = initB;
    this.mxLearnStep = mxLearnStep;
  }

  public CodingMatrixLearning(final Mx initB, final double mxLearnStep) {
    this(initB, mxLearnStep, initB.rows(), 1.0, initB.rows());
  }

  public CodingMatrixLearning(final int k, final int l, final double lambdaC, final double lambdaR, final double lambda1, final double mxLearnStep) {
    this(new VecBasedMx(k, l), mxLearnStep, lambdaC, lambdaR, lambda1);
    final Random rand = new FastRandom(100500);
    do {
      for (int i = 0; i < k; i++) {
        for (int j = 0; j < l; j++) {
          initB.set(i, j, rand.nextInt(3) - 1);
        }
      }
    } while (!CMLHelper.checkConstraints(initB));
  }

  public CodingMatrixLearning(final int k, final int l, final double mxLearnStep) {
    this(k, l, k, 1.0, k, mxLearnStep);
  }

  @Override
  public Mx findMatrixB(final Mx S) {
    Mx mxB = initB;

    final Vec b = new ArrayVec(2*k*l + 2*l + k);
    {
      for (int i = 0; i < 2*k*l; i++)
        b.set(i, 1.);
      for (int i = 2* k * l; i < 2*k*l + 2*l; i++)
        b.set(i, -2.);
      for (int i = 2* k * l + 2* l; i < 2*k*l + 2*l + k; i++)
        b.set(i, -1.);
    }

    final Mx Inv = new VecBasedMx(k, k);
    {
      final double mult = 1 / (k * lambdaR * lambdaC + lambdaC * lambdaC);
      VecTools.fill(Inv, -lambdaR * mult);
      for (int i = 0; i < Inv.columns(); i++)
        Inv.adjust(i, i, (k * lambdaR + lambdaC) * mult);
      VecTools.scale(Inv, 0.5); //see algorithm's iteration process
    }

    final Vec gamma = new ArrayVec(2*k*l + 2*l + k);
    {
//      init gamma
      for (int i = 0; i < gamma.dim(); i++) {
        gamma.set(i, 0.5);
      }
    }

    final Vec mu = new ArrayVec(k*l);
    {
//      init mu
      for (int i = 0; i < mu.dim(); i++) {
        mu.set(i, lambda1 / 2);
      }
    }

    int iter = 0;
    double error = 100500;
    while (error > MX_LEARN_EPS) {
      /**
       * B^{i+1} = Inv * (2S * B^{i} - (transpose(A) * gamma - mu))
       * def: m1 = 2S * B^{i}
       *      m2 = transpose(A) * gamma
       *      sub1 = m2 - mu
       *      sub2 = m1 - Mx(sub1)
       */

      final Mx A = createConstraintsMatrix(mxB);
      {
        final Mx m1 = MxTools.multiply(S, mxB);
        VecTools.scale(m1, 2.);
        final Vec m2 = MxTools.multiply(MxTools.transpose(A), gamma);
        final Vec sub1 = VecTools.subtract(m2, mu);
        final Mx sub1Mx = vec2mx(sub1, m1.columns());
        final Mx sub2 = VecTools.subtract(m1, sub1Mx);
        final Mx newMxB = MxTools.multiply(Inv, sub2);
        error = VecTools.infNorm(VecTools.subtract(mxB, newMxB));
        mxB = newMxB;
      }

      /**
       * Projections:
       * gamma = Pr_{gamma >= 0} (gamma - t * (b - A * vec(mxB)))
       * def: m1 = A * vec(mxB)
       *      sub = b - m1
       *
       * mu = Pr_{infnorm(mu) <= lambda1} (mu - t * vec(mxB))
       */
      {
        final Vec vecB = mx2vec(mxB);
        final Vec m1 = MxTools.multiply(A, vecB);
        final Vec sub = VecTools.subtract(b, m1);
        VecTools.incscale(gamma, sub, -1 * mxLearnStep);
        for (final VecIterator iterator = gamma.nonZeroes(); iterator.advance(); ) {
          if (iterator.value() < 0)
            iterator.setValue(0);
        }

        VecTools.incscale(mu, vecB, -1 * mxLearnStep);
        for (final VecIterator iterator = mu.nonZeroes(); iterator.advance(); ) {
          if (Math.abs(iterator.value()) > lambda1) {
            iterator.setValue(lambda1);
          }
        }
      }
      if (iter++ > 1000)
        break;

//      if (!checkConstraints(mxB))
//        throw new IllegalStateException("out of contraints!");

    }
    normalizeMx(mxB);
    return mxB;
  }

  private static void normalizeMx(final Mx codingMatrix) {
    for (final MxIterator iter = codingMatrix.nonZeroes(); iter.advance(); ) {
      final double value = iter.value();
      if (Math.abs(value) > MX_IGNORE_THRESHOLD)
        iter.setValue(Math.signum(value));
      else
        iter.setValue(0.0);
    }
  }

  protected static Mx vec2mx(final Vec vec, final int columns) {
    final Mx result = new VecBasedMx(columns, new ArrayVec(vec.dim()));
    final int rows = result.rows();
    for (int i = 0; i < vec.dim(); i++) {
      result.set(i % rows, i / rows, vec.get(i));
    }
    return result;
  }

  protected static Vec mx2vec(final Mx mx) {
    final Vec result = new ArrayVec(mx.dim());
    final int rows = mx.rows();
    for (int i = 0; i < result.dim(); i++) {
      result.set(i, mx.get(i % rows, i / rows));
    }
    return result;
  }
//
//  public static boolean checkConstraints(final Mx B) {
//    final int k = B.rows();
//    final int l = B.columns();
//    final Mx A = createConstraintsMatrix(B);
//    final Vec vecB = mx2vec(B);
//    final Vec checkVec = VecTools.multiply(A, vecB);
//    for (int i = 0; i < 2*k*l; i++)
//      if (checkVec.at(i) > 1.)
//        return false;
//    for (int i = 2* k * l; i < 2*k*l + 2*l; i++)
//      if (checkVec.at(i) > -2.)
//        return false;
//    for (int i = 2* k * l + 2* l; i < 2*k*l + 2*l + k; i++)
//      if (checkVec.at(i) > -1)
//        return false;
//    return true;
//  }


  /**
   *
   * @param B Coding matrix that was obtained at the last iteration, size = [k,l]
   * @return Matrix of constraints
   */
  public static Mx createConstraintsMatrix(final Mx B) {
    final int k = B.rows();
    final int l = B.columns();

//    final Mx A = new SparseMx<MxBasisImpl>(new MxBasisImpl(2*k*l + 2*l +k, k*l));
    final Mx A = new VecBasedMx(2* k * l + 2* l + k, k * l);
    for (int j = 0; j < k * l; j++) {
      A.set(j, j, -1.0);
      A.set(k * l + j, j, 1.0);
      final double signum = Math.signum(B.get(j % k, j / k));
      A.set(2*k*l + j/ k, j, -1 - signum);
      A.set(2*k*l + l + j/ k, j, 1 -signum);
      A.set(2*k*l + 2*l + (j % k), j, -signum);
    }
    return A;
  }
}