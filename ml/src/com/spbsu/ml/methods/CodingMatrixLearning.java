package com.spbsu.ml.methods;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecIterator;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.commons.math.vectors.impl.VecArrayMx;
import com.spbsu.commons.math.vectors.impl.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.VecBasedMxCP;
import com.spbsu.ml.Trans;
import com.spbsu.ml.data.DSIterator;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.loss.LLLogit;
import gnu.trove.iterator.TIntIterator;
import gnu.trove.list.TIntList;
import gnu.trove.list.linked.TIntLinkedList;
import gnu.trove.map.TIntObjectMap;
import gnu.trove.map.hash.TIntIntHashMap;
import gnu.trove.map.hash.TIntObjectHashMap;

/**
 * User: qdeee
 * Date: 07.05.14
 */
public class CodingMatrixLearning implements Optimization<LLLogit> {
  private final int k;
  private final int l;

  public CodingMatrixLearning(final int k, final int l) {
    this.k = k;
    this.l = l;
  }

  @Override
  public Trans fit(final DataSet learn, final LLLogit llLogit) {

    return null;
  }

  public Mx createSimilarityMatrix(DataSet learn) {
    final TIntObjectMap<TIntList> indexes = new TIntObjectHashMap<TIntList>();
    for (DSIterator iter = learn.iterator(); iter.advance(); ) {
      final int catId = (int) iter.y();
      if (indexes.containsKey(catId)) {
        indexes.get(catId).add(iter.index());
      }
      else {
        final TIntList newClassIdxs = new TIntLinkedList();
        newClassIdxs.add(iter.index());
        indexes.put(catId, newClassIdxs);
      }
    }

    final Mx S = new VecBasedMx(k, k);
    for (int i = 0; i < k; i++) {
      final TIntList classIdxsI = indexes.get(i);
      for (int j = i; j < k; j++) {
        final TIntList classIdxsJ = indexes.get(j);
        double value = 0.;
        for (TIntIterator iterI = classIdxsI.iterator(); iterI.hasNext(); ) {
          final int i1 = iterI.next();
          for (TIntIterator iterJ = classIdxsJ.iterator(); iterJ.hasNext(); ) {
            final int i2 = iterJ.next();
            value += VecTools.distance(learn.data().row(i1), learn.data().row(i2));
          }
        }
        value /= classIdxsI.size() * classIdxsJ.size();
        S.set(i, j, value);
        S.set(j, i, value);
      }
    }
    return S;

  }

  protected static Mx vec2mx(final Vec vec, int columns) {
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

  public Mx findMatrixB(final Mx S, int iters, double step, double lambdaC, double lambdaR, double lambda1) {
    Mx mxB = new VecBasedMx(l, new ArrayVec(k*l));
    {
      //init B
    }

    final Vec b = new ArrayVec(2* k * l + 2* l + k);
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
      VecTools.fill(Inv, -lambdaR / mult);
      for (int i = 0; i < Inv.columns(); i++)
        Inv.adjust(i, i, (k * lambdaR + lambdaC) / mult);
      VecTools.scale(Inv, 0.5); //see algorithm's iteration process
    }

    final Vec gamma = new ArrayVec(2*k*l + 2*l + k);
    {
      //init gamma
      for (int i = 0; i < gamma.dim(); i++) {
        gamma.set(i, 0.5);
      }
    }

    final Vec mu = new ArrayVec(k*l);
    {
      //init mu
      for (int i = 0; i < mu.dim(); i++) {
        mu.set(i, lambda1 / 2);
      }
    }

    int iter = 0;
    while (iter++ < iters) {
      /**
       * B^{i+1} = Inv * (2S * B^{i} - (transpose(A) * gamma - mu))
       * def: m1 = 2S * B^{i}
       *      m2 = transpose(A) * gamma
       *      sub1 = m2 - mu
       *      sub2 = m1 - Mx(sub1)
       */

      final Mx A = createConstraintsMatrix(mxB);
      {
        final Mx m1 = VecTools.multiply(S, mxB);
        VecTools.scale(m1, 2.);
        final Vec m2 = VecTools.multiply(VecTools.transpose(A), gamma);
        final Vec sub1 = VecTools.subtract(m2, mu);
        final Mx sub1Mx = vec2mx(sub1, m1.columns());
        final Mx sub2 = VecTools.subtract(m1, sub1Mx);
        mxB = VecTools.multiply(Inv, sub2);
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
        final Vec m1 = VecTools.multiply(A, vecB);
        final Vec sub = VecTools.subtract(b, m1);
        VecTools.incscale(gamma, sub, -1 * step);
        for (VecIterator iterator = gamma.nonZeroes(); iterator.advance(); ) {
          if (iterator.value() < 0)
            iterator.setValue(0);
        }

        VecTools.incscale(mu, vecB, -1 * step);
        for (VecIterator iterator = mu.nonZeroes(); iterator.advance(); ) {
          if (Math.abs(iterator.value()) > lambda1) {
            iterator.setValue(lambda1);
          }
        }
      }
      System.out.println(mxB.toString());
      System.out.println();

    }
    return mxB;
  }

  /**
   *
   * @param B Coding matrix that was obtained at the last iteration, size = [k,l]
   * @return Matrix of constraints
   */
  public Mx createConstraintsMatrix(final Mx B) {
    final Mx A = new VecBasedMx(2* k * l + 2* l + k, k * l);
    for (int j = 0; j < k * l; j++) {
      A.set(j, j, -1.0);
      A.set(k * l + j, j, 1.0);
      System.out.println(j % k + " " + j / k);
      final double signum = Math.signum(B.get(j % k, j / k));
      A.set(2* k * l + j/ k, j, -1 - signum);
      A.set(2* k * l + l + j/ k, j, 1 -signum);
      A.set(2* k * l + 2* l + j% k, j, -signum);
    }
    return A;
  }


}
