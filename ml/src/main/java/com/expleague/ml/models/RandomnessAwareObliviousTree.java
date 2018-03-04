package com.expleague.ml.models;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.ml.FeatureBinarization;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.distributions.DistributionConvolution;
import com.expleague.ml.distributions.RandomVariable;
import com.expleague.ml.distributions.RandomVec;
import com.expleague.ml.distributions.parametric.DeltaFunction;
import com.expleague.ml.distributions.parametric.NormalDistributionImpl;
import com.expleague.ml.randomnessAware.RandomFunc;

import java.util.Arrays;
import java.util.List;


/**
 * User: noxoomo
 */
public class RandomnessAwareObliviousTree extends RandomFunc.Stub implements RandomFunc {
  private static final double eps = 1e-15;
  private final FeatureBinarization.BinaryFeature[] splits;
  private final RandomVariable[] values;

  public RandomnessAwareObliviousTree(final List<FeatureBinarization.BinaryFeature> splits,
                                      final double[] vals,
                                      DistributionConvolution convolution) {
    super(convolution);
    this.splits = splits.toArray(new FeatureBinarization.BinaryFeature[splits.size()]);
    if (this.splits.length == 0)
      throw new RuntimeException("Creating oblivious tree of zero depth");
    this.values = new RandomVariable[vals.length];
    for (int i = 0; i < vals.length; ++i) {
      final int finalI = i;
      this.values[i] = new NormalDistributionImpl(vals[finalI], 0);
    }
  }

  public RandomnessAwareObliviousTree(final List<FeatureBinarization.BinaryFeature> splits,
                                      final RandomVariable[] vals,
                                      DistributionConvolution convolution) {
    super(convolution);
    this.splits = splits.toArray(new FeatureBinarization.BinaryFeature[splits.size()]);
    if (this.splits.length == 0)
      throw new RuntimeException("Creating oblivious tree of zero depth");
    this.values = vals;
  }


  @Override
  public RandomVariable appendTo(final double scale, final Vec feature, final RandomVariable to) {
    double[] probs = new double[splits.length];
    for (int i = 0; i < splits.length; ++i) {
      RandomVariable rv = splits[i].featureBinarization().featureExtractor().compute(feature);
      if (splits[i] instanceof FeatureBinarization.TakeEqualFeature) {
        final double value;
        if (rv instanceof DeltaFunction) {
          value = ((DeltaFunction) rv).value();
        }
        else {
          throw new RuntimeException("wrong type");
        }
        probs[i] = splits[i].value(value) ? 1 : 0;
      }
      else {
        probs[i] = 1.0 - rv.cdf(splits[i].border());
      }
    }

    double totalProb = 0;
    for (int i = 0; i < values.length; ++i) {
      double p = 1.0;
      for (int j = 0; j < splits.length; ++j) {
        final int bit = 1 << (splits.length - j - 1);
        p *= (i & bit) != 0 ? probs[j] : 1.0 - probs[j];
        if (p == 0) {
          break;
        }
      }
      if (p > eps) {
        convolution.combine(values[i], scale * p, to, 1.0);
      }
      totalProb += p;
    }
    assert (Math.abs(totalProb - 1.0) < 1e-8);
    return to;
  }

  @Override
  public RandomVec appendTo(final double scale, final VecDataSet dataSet, final RandomVec dst) {

    final RandomVec features[] = new RandomVec[splits.length];
    for (int depth = 0; depth < splits.length; ++depth) {
      features[depth] = splits[depth].featureBinarization().featureExtractor().computeAll(dataSet);
    }
    for (int doc = 0; doc < dataSet.data().rows(); ++doc) {
      double[] probs = new double[splits.length];

      for (int i = 0; i < splits.length; ++i) {
        RandomVariable rv = features[i].at(doc);
        if (splits[i] instanceof FeatureBinarization.TakeEqualFeature) {
          double value;
          if (rv instanceof DeltaFunction) {
            value = ((DeltaFunction) rv).value();
          } else {
            throw new RuntimeException("err");
          }
          probs[i] = splits[i].value(value) ? 1 : 0;
        }else {
//          P(rv > splits[i].border()) = 1.0  - P(rv <= splits[i].border())
          probs[i] = 1.0 - rv.cdf(splits[i].border());
//          if (splits[i].value(rv.expectation()) == (probs[i] == 1.0)) {
//            throw new RuntimeException("Bug");
//          }
        }
      }
//      convolution.combine(values[bin], scale, doc, dst, 1.0);


      for (int i = 0; i < values.length; ++i) {
        double p = 1.0;
        for (int j = 0; j < splits.length; ++j) {
          final int bit = 1 << (splits.length - j - 1);
          p *= (i & bit) != 0 ? probs[j] : 1.0 - probs[j];
        }
        if (p != 0) {
          convolution.combine(values[i], scale * p, doc, dst, 1.0);
        }
//        totalProb += p;
      }
//      assert (Math.abs(totalProb - 1.0) < 1e-8);
    }
    return dst;
  }

  public int dim() {
    return splits[0].featureBinarization().featureExtractor().dim();
  }

  @Override
  public String toString() {
    final StringBuilder builder = new StringBuilder();
    builder.append(values.length);
    builder.append("->(");
    for (int i = 0; i < splits.length; i++) {
      builder.append(i > 0 ? ", " : "")
          .append(splits[i]);
    }
    builder.append(")");
    builder.append("+[");
    for (final RandomVariable feature : values) {
      builder.append(feature).append(", ");
    }
    builder.delete(builder.length() - 2, builder.length());
    builder.append("]");
    return builder.toString();
  }


  @Override
  public boolean equals(final Object o) {
    if (this == o) return true;
    if (!(o instanceof RandomnessAwareObliviousTree)) return false;

    final RandomnessAwareObliviousTree that = (RandomnessAwareObliviousTree) o;

    if (!Arrays.equals(splits, that.splits)) return false;
    if (!Arrays.equals(values, that.values)) return false;

    return true;
  }

  @Override
  public int hashCode() {
    int result = Arrays.hashCode(splits);
    result = 31 * result + Arrays.hashCode(values);
    return result;
  }
}
