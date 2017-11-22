package com.expleague.ml.models;

import com.expleague.commons.math.Func;
import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.SingleValueVec;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.random.FastRandom;
import com.expleague.commons.util.ArrayTools;
import com.expleague.ml.ComputeBinarization;
import com.expleague.ml.ComputeBinarizedFeature;
import com.expleague.ml.FeatureBinarization;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.distributions.RandomVariable;
import com.expleague.ml.distributions.RandomVec;
import com.expleague.ml.distributions.parametric.DeltaFunction;
import com.expleague.ml.distributions.samplers.RandomVariableSampler;
import com.expleague.ml.distributions.samplers.RandomVecSampler;
import com.expleague.ml.randomnessAware.*;

import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;
import java.util.stream.Stream;


/**
 * User: noxoomo
 */
public class RandomnessAwareObliviousTree extends RandomnessAwareTrans.Stub<BinOptimizedRandomnessPolicy> implements RandomnessAwareTrans<BinOptimizedRandomnessPolicy>, Func {
  private final FeatureBinarization.BinaryFeature[] splits;
  private final RandomVariable[] values;
  private final RandomVariableSampler[] sampler;


  public RandomnessAwareObliviousTree(final List<FeatureBinarization.BinaryFeature> splits,
                                      final double[] vals) {
    super(BinOptimizedRandomnessPolicy.SampleBin);
    this.splits = splits.toArray(new FeatureBinarization.BinaryFeature[splits.size()]);
    if (this.splits.length == 0)
      throw new RuntimeException("Creating oblivious tree of zero depth");
    this.values = new RandomVariable[vals.length];
    for (int i = 0; i < vals.length; ++i) {
      final int finalI = i;
      this.values[i] = (DeltaFunction) () -> vals[finalI];
    }
    sampler = createSampler();
  }

  private RandomVariableSampler[] createSampler() {
    final RandomVariableSampler[] samplers = new RandomVariableSampler[values.length];
    for (int i = 0; i < samplers.length; ++i) {
      samplers[i] = values[i].sampler();
    }
    return samplers;
  }

  public RandomnessAwareObliviousTree(final List<FeatureBinarization.BinaryFeature> splits,
                                      final RandomVariable[] vals) {
    super(BinOptimizedRandomnessPolicy.SampleBin);
    this.splits = splits.toArray(new FeatureBinarization.BinaryFeature[splits.size()]);
    if (this.splits.length == 0)
      throw new RuntimeException("Creating oblivious tree of zero depth");
    this.values = vals;
    sampler = createSampler();
  }

  public int dim() {
    return splits[0].owner().owner().dim();
  }

//  public double value(final Vec x) {
//    return transAll(x, activePolicy);
//    if (activePolicy.type() == ProcessRandomnessPolicy.Type.SampleBin) {
//      int index = 0;
//      for (int i = 0; i < splits.length; ++i) {
//        if (splits[i].value(x, activePolicy) > 0) {
//          index |= 1 << i;
//        }
//      }
//      return values[index];
//    } else {
//
//    }
//  }

  public Mx transAll(final VecDataSet dataSet) {
    final Mx result = new VecBasedMx(dataSet.length(), 1);

    final BinOptimizedRandomnessPolicy policy = activePolicy();

    final RandomVec<?> features[] = new RandomVec[splits.length];
    final RandomVecSampler samplers[] = new RandomVecSampler[splits.length];

    for (int depth = 0; depth < splits.length; ++depth) {
      features[depth] = splits[depth].owner().owner().apply(dataSet);
      samplers[depth] = policy == BinOptimizedRandomnessPolicy.SampleBin ? features[depth].sampler() : null;
    }

    switch (policy) {
      case SampleBin:
      case PointEstimateBin: {
        final FastRandom random = random();

        IntStream.range(0, result.rows()).parallel().forEach(doc -> {
//        for (int doc = 0; doc < result.rows(); ++doc) {
          int bin = 0;
          for (int depth = 0; depth < splits.length; ++depth) {
            final double featureInstance = policy == BinOptimizedRandomnessPolicy.SampleBin
                ? samplers[depth].instance(random, doc)
                : features[depth].expectation(doc); //point estimate bin
            bin <<= 1;
            if (splits[depth].value(featureInstance)) {
              ++bin;
            }
            final double instance = policy == BinOptimizedRandomnessPolicy.SampleBin ? sampler[bin].instance(random) : values[bin].mean();
            result.set(doc, instance);
          }
        });
        return result;
      }
      case BinsExpectation: {
        double[] probs = new double[splits.length];

        for (int doc = 0; doc < result.rows(); ++doc) {

          for (int i = 0; i < splits.length; ++i) {
            RandomVariable rv = features[i].randomVariable(doc);
            if (splits[i] instanceof FeatureBinarization.TakeEqualFeature) {
              probs[i] = splits[i].value(rv.mean()) ? 1 : 0;
            }
            else {
              probs[i] = 1.0 - rv.cdf(splits[i].border());
            }
          }

          double res = 0.0;
          double totalProb = 0;
          for (int i = 0; i < values.length; ++i) {
            double p = 1.0;
            for (int j = 0; j < splits.length; ++j) {
              final int bit = 1 << (splits.length - j - 1);
              p *= (i & bit) != 0 ? probs[j] : 1.0 - probs[j];
            }
            res += p * values[i].mean();
            totalProb += p;
          }
          assert (Math.abs(totalProb - 1.0) < 1e-8);
          result.set(doc, res);
        }
        return result;
      }
      default: {
        throw new RuntimeException("unimplemented");
      }
    }
  }

  @Override
  public final Vec trans(final Vec x) {
    return new SingleValueVec(value(x));
  }


  public double value(final Vec feature) {
    final BinOptimizedRandomnessPolicy policy = activePolicy();
    switch (policy) {
      case PointEstimateBin:
      case SampleBin: {
        int index = 0;
        final FastRandom random = random();

        for (int i = 0; i < splits.length; ++i) {
          RandomVariable rv = splits[i].owner().owner().compute(feature);
          final double value = policy == BinOptimizedRandomnessPolicy.SampleBin ? rv.sampler().instance(random) : rv.mean();
          index <<= 1;
          if (splits[i].value(value)) {
            ++index;
          }
        }
        return sampler[index].instance(random);
      }
      case BinsExpectation: {
        double[] probs = new double[splits.length];
        for (int i = 0; i < splits.length; ++i) {
          RandomVariable rv = splits[i].owner().owner().compute(feature);
          if (splits[i] instanceof FeatureBinarization.TakeEqualFeature) {
            probs[i] = splits[i].value(rv.mean()) ? 1 : 0;
          }
          else {
            probs[i] = 1.0 - rv.cdf(splits[i].border());
          }
        }

        double result = 0.0;
        double totalProb = 0;
        for (int i = 0; i < values.length; ++i) {
          double p = 1.0;
          for (int j = 0; j < splits.length; ++j) {
            final int bit = 1 << (splits.length - j - 1);
            p *= (i & bit) != 0 ? probs[j] : 1.0 - probs[j];
          }
          result += p * values[i].mean();
          totalProb += p;
        }
        assert (Math.abs(totalProb - 1.0) < 1e-8);
        return result;
      }
      default: {
        throw new RuntimeException("unimplemented");
      }
//      double[] probs = new double[values.length];
//      Arrays.fill(probs, 1.0f);
//
//      double result = 0.0;
//      double totalProb = 0;
//
//      for (int i = 0; i < values.length; ++i) {
//        double p = 1.0;
//        for (int j = 0; j < splits.length; ++j) {
//          p *= ((i >> j) & 1) != 0 ? probs[j] : 1.0 - probs[j];
//        }
//        result += p * values[i].mean();
//        totalProb += p;
//      }
//      assert (Math.abs(totalProb - 1.0) < 1e-8);
//      return result;
    }
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

  @Override
  public final int ydim() {
    return 1;
  }

  @Override
  public final int xdim() {
    return dim();
  }


  static class ObliviousTreePolicy implements ProcessRandomnessPolicy {
    BinOptimizedRandomnessPolicy binsPolicy;
    LeavesPolicy leavesPolicy;

    enum LeavesPolicy {
      Sample,
      Expectation
    }
  }
}
