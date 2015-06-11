package com.spbsu.ml.methods.rvm;

import com.spbsu.commons.math.stat.StatTools;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.MxTools;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.ml.func.BiasedLinear;
import gnu.trove.iterator.TIntIterator;
import org.apache.commons.math3.util.FastMath;

/**
 * Created by noxoomo on 01/06/15.
 */

//Relevance vector machine model
// target = Xw + \varepsilon, \varepsilon \sim N(0,\sigma^2), w_i \sim N(0, 1/a_i)
// X  — set of basis functions
// see bishop pattern recognition and machine learning or tipping articles for details
class RVMCache {
  final Mx data;
  final Vec target;

  final DotProductsCache featureProducts;

  final private double[] alpha; //precision prior for weights
  final private double[] q;
  final private double[] s;
  final private double[] theta;
  final private double[] diffs;

  double noiseVariance;

  final ActiveIndicesSet activeIndices;

  private Mx sigma;
  private Vec mu;
  private Vec predictions;


  RVMCache(final Mx data, final Vec target, final FastRandom random) {
    this.data = data;
    this.target = target;
    this.predictions = new ArrayVec(target.dim());
    this.alpha = new double[data.columns() + 1]; //last component for bias
    this.s = new double[data.columns() + 1];
    this.q = new double[data.columns() + 1];
    this.theta = new double[data.columns() + 1];
    this.diffs = new double[data.columns() + 1];

    noiseVariance = StatTools.variance(target) / 4;
    final double w = target.dim();
    alpha[data.columns()] = w / (VecTools.sum2(target) / w - noiseVariance);
    featureProducts = new DotProductsCache(data, target);
    activeIndices = new ActiveIndicesSet(this.alpha.length, random);
    activeIndices.addToActive(data.columns());

    for (int i = 0; i < data.columns(); ++i)
      alpha[i] = Double.POSITIVE_INFINITY;

    estimateVariance();
  }

  void estimateVariance() {
    calcSigma();
    calcMean();
    updatePredictions();
    noiseVariance = calcNoiseVariance();
    updateQuantities();
  }

  private double calcNoiseVariance() {
    double denum = target.dim() - activeIndices.size();
    TIntIterator indices = activeIndices.activeIterator();
    {
      int i = 0;
      while (indices.hasNext()) {
        int index = indices.next();
        denum += alpha[index] * sigma.get(i++);
      }
    }
    return VecTools.l2(predictions, target) / denum;
  }


  void calcSigma() {
    Mx invSigma = new VecBasedMx(activeIndices.size(), activeIndices.size());
    int[] indices = activeIndices.activeIndices();
    for (int i = 0; i < indices.length; ++i) {
      final int firstIndex = indices[i];
      invSigma.adjust(i, i, alpha[firstIndex]);
      invSigma.adjust(i, i, featureProducts.featuresProduct(firstIndex, firstIndex) / noiseVariance);
      for (int j = i + 1; j < indices.length; ++j) {
        final int secondIndex = indices[j];
        final double val = featureProducts.featuresProduct(firstIndex, secondIndex) / noiseVariance;
        invSigma.adjust(i, j, val);
        invSigma.adjust(j, i, val);
      }
    }
    sigma = MxTools.inverse(invSigma);
  }

  void calcMean() {
    Vec result = new ArrayVec(activeIndices.size());
    TIntIterator index = activeIndices.activeIterator();
    int i = 0;
    while (index.hasNext()) {
      result.adjust(i++, featureProducts.targetProducts(index.next()));
    }
    result = MxTools.multiply(sigma, result);
    VecTools.scale(result, 1.0 / noiseVariance);
    mu = result;
  }

  void updatePredictions() {
    for (int point = 0; point < target.dim(); ++point) {
      TIntIterator indices = activeIndices.activeIterator();
      int i = 0;
      double result = 0;
      while (indices.hasNext()) {
        final int ind = indices.next();
        result += ind < data.columns() ? mu.get(i++) * data.get(point, ind) : mu.get(i++);
      }
      predictions.set(point, result);
    }
  }


  void updateQuantities() {
    calcSigma();
    calcMean();
    for (int i = 0; i < alpha.length; ++i) {
      updateQuantities(i);
    }


//    {
//      Mx A = new VecBasedMx(alpha.length,alpha.length);
//      for (int i=0; i < alpha.length;++i) {
//        A.set(i,i,1.0 / alpha[i]);
//      }
//
//      Mx C = MxTools.multiply(MxTools.multiply(F,A),trF);
//      assert(C.columns() == C.rows());
//      for (int i=0; i < C.columns();++i) {
//        C.adjust(i,i,noiseVariance);
//      }
//
////      Vec S = new ArrayVec(F.columns());
////      Vec Q = new ArrayVec(F.columns());
//      Mx invC = MxTools.inverseCholesky(C);
//      Vec cTar = MxTools.multiply(invC, target);
//      for (int i=0; i < F.columns();++i){
//        s[i] =  VecTools.multiply(F.col(i), MxTools.multiply(invC, F.col(i)));
//        q[i] =  VecTools.multiply(F.col(i), cTar);
//      }
//
////      for (int i=0; i < S.dim();++i) {
////        s[i] = S.get(i);
////        q[i] = Q.get(i);
//////        if (Math.abs(S.get(i)- s[i]) > 1e-2) {
//////          System.out.println(i);
//////        }
////////        assert(Math.abs(S.get(i)- s[i]) < 1e-3);
//////
//////        if (Math.abs(Q.get(i)- q[i]) > 1e-2) {
//////          System.out.println(i);
//////        }
//////        assert(Math.abs(Q.get(i)- q[i]) < 1e-3);
////      }
//    }
  }

  private void updateQuantities(int feature) {
    Vec dotFeatureWithActiveFeatures = new ArrayVec(activeIndices.size());
    {
      TIntIterator activeFeatures = activeIndices.activeIterator();
      int i = 0;
      while (activeFeatures.hasNext()) {
        final int ind = activeFeatures.next();
        final double res = featureProducts.featuresProduct(feature, ind) / noiseVariance;
        dotFeatureWithActiveFeatures.set(i++, res);
      }
    }

    s[feature] = featureProducts.featuresProduct(feature, feature) / noiseVariance
      - VecTools.multiply(dotFeatureWithActiveFeatures, MxTools.multiply(sigma, dotFeatureWithActiveFeatures));
    q[feature] = featureProducts.targetProducts(feature) / noiseVariance -
      VecTools.multiply(dotFeatureWithActiveFeatures, mu);
  }

  enum Result {
    Remove,
    Add,
    Updated,
    Skipped
  }

  public Result update(int i) {
    final double si = Double.isInfinite(alpha[i]) ? s[i] : alpha[i] * s[i] / (alpha[i] - s[i]);
    final double qi = Double.isInfinite(alpha[i]) ? q[i] : alpha[i] * q[i] / (alpha[i] - s[i]);
    theta[i] = qi * qi - si;
    if (theta[i] > 0) {
      final double oldAlpha = alpha[i];
      alpha[i] = si * si / theta[i];
      if (Double.isInfinite(oldAlpha)) {
        activeIndices.addToActive(i);
        diffs[i] = Double.POSITIVE_INFINITY;
        return Result.Add;
      } else {
        diffs[i] = FastMath.abs(FastMath.log(oldAlpha) - FastMath.log(alpha[i]));
        return Result.Updated;
      }
    } else if (theta[i] < 0 && Math.abs(alpha[i]) <= Double.MAX_VALUE) {
      alpha[i] = Double.POSITIVE_INFINITY;
      diffs[i] = 0;
      activeIndices.removeFromActive(i);
      return Result.Remove;
    }
    return Result.Skipped;
  }

  private boolean stop(double tolerance) {
    for (int i = 0; i < diffs.length; ++i) {
      if (diffs[i] > tolerance || (Double.isInfinite(alpha[i]) && theta[i] > 1e-3))
        return false;
    }
    return true;
  }

  public BiasedLinear fit(double tolerance) {
    do {
      TIntIterator nextFeature = activeIndices.indicesIterator();
      while (nextFeature.hasNext()) {
        if (update(nextFeature.next()) != Result.Skipped)
          updateQuantities();
      }
      estimateVariance();
    } while (!stop(tolerance));

    final double[] weights = new double[data.columns()];
    double bias = 0;
    {
      TIntIterator active = activeIndices.activeIterator();
      int i = 0;
      while (active.hasNext()) {
        final int ind = active.next();
        if (ind != data.columns()) {
          weights[ind] = mu.get(i++);
        } else {
          bias = mu.get(i++);
        }
      }
    }
    return new BiasedLinear(weights, bias);
  }


  //lazy cache for dot products
  //TODO: merge with elastic net
  //TODO: reduce x2 memory usage
  //not thread safe
  static class DotProductsCache {
    final Vec target;
    final Mx data;

    private final boolean[] isFeaturesProductCached;
    private final boolean[] isTargetCached;
    private final Mx featureProducts;
    private final Vec targetProducts;

    public DotProductsCache(final Mx data, final Vec target) {
      this.target = target;
      this.data = data;
      this.isFeaturesProductCached = new boolean[(data.columns() + 1) * (data.columns() + 1)];
      this.isTargetCached = new boolean[data.columns() + 1];
      this.featureProducts = new VecBasedMx(data.columns() + 1, data.columns() + 1);
      this.targetProducts = new ArrayVec(data.columns() + 1);
    }

    public double featuresProduct(int i, int j) {
      if (isFeaturesProductCached[i * featureProducts.columns() + j]) {
        return featureProducts.get(i, j);
      } else {
        final double v;
        if (i != data.columns() && j != data.columns()) {
          v = VecTools.multiply(data.col(i), data.col(j));
        } else {
          if (i == j) {
            v = data.rows();
          } else {
            int ind = i < j ? i : j;
            v = VecTools.sum(data.col(ind));
          }
        }
        featureProducts.set(i, j, v);
        featureProducts.set(j, i, v);
        isFeaturesProductCached[i * featureProducts.columns() + j] = true;
        isFeaturesProductCached[j * featureProducts.columns() + i] = true;
        return v;
      }
    }

    public double targetProducts(int i) {
      if (isTargetCached[i]) {
        return targetProducts.get(i);
      } else {
        final double v;
        if (i != data.columns()) {
          v = VecTools.multiply(data.col(i), target);
        } else {
          v = VecTools.sum(target);
        }
        targetProducts.set(i, v);
        isTargetCached[i] = true;
        return v;
      }
    }
  }

  static class ActiveIndicesSet {
    private final int[] set;
    private final int[] indicesMap;
    private int cursor;
    private final FastRandom random;

    public ActiveIndicesSet(int size, FastRandom rand) {
      set = ArrayTools.sequence(0, size);
      cursor = 0;
      indicesMap = ArrayTools.sequence(0, size);
      this.random = rand;
    }

    public TIntIterator indicesIterator() {
      return new TIntIterator() {
        int current = 0;

        @Override
        public int next() {
//          return current++;
          ++current;
          return random.nextInt(set.length);
        }

        @Override
        public boolean hasNext() {
          return current < 10*set.length;
        }

        @Override
        public void remove() {
          if (indicesMap[current] < cursor) {
            removeFromActive(current);
          }
        }
      };
    }

    public void addToActive(int index) {
      if (indicesMap[index] < cursor || index >= indicesMap.length) {
        throw new IllegalArgumentException("already active index");
      }
      swapWithCursor(index);
      ++cursor;
    }

    public int size() {
      return cursor;
    }

    private void swapWithCursor(int index) {
      final int movedInd = set[cursor];
      final int movedTo = indicesMap[index];
      indicesMap[index] = cursor;
      set[cursor] = index;
      indicesMap[movedInd] = movedTo;
      set[movedTo] = movedInd;
    }

    public void removeFromActive(int index) {
      --cursor;
      swapWithCursor(index);
    }

    TIntIterator activeIterator() {
      return new TIntIterator() {
        int current = 0;

        @Override
        public int next() {
          return set[current++];
        }

        @Override
        public boolean hasNext() {
          return current < cursor;
        }

        @Override
        public void remove() {
          throw new UnsupportedOperationException("unsupported");
        }
      };
    }

    int[] activeIndices() {
      int[] result = new int[cursor];
      System.arraycopy(set, 0, result, 0, result.length);
      return result;
    }
  }

}
