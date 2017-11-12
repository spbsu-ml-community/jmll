package com.expleague.ml.models;

import com.expleague.commons.math.Func;
import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.SingleValueVec;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.random.FastRandom;
import com.expleague.ml.FeatureBinarization;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.distributions.RandomVariable;
import com.expleague.ml.distributions.RandomVec;
import com.expleague.ml.distributions.parametric.DeltaFunction;
import com.expleague.ml.distributions.samplers.RandomVariableSampler;
import com.expleague.ml.distributions.samplers.RandomVecSampler;
import com.expleague.ml.randomnessAware.ProcessRandomnessPolicy;
import com.expleague.ml.randomnessAware.RandomnessAwareTrans;

import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;


/**
 * User: noxoomo
 */
public class RandomnessAwareRegion extends RandomnessAwareTrans.Stub<BinOptimizedRandomnessPolicy> implements RandomnessAwareTrans<BinOptimizedRandomnessPolicy>, Func {
  private final FeatureBinarization.BinaryFeature[] splits;
  private final boolean[] masks;
  private final RandomVariable[] values;
  private final RandomVariableSampler[] sampler;


  public RandomnessAwareRegion(final List<FeatureBinarization.BinaryFeature> splits,
                               final List<Boolean> masks,
                               final double[] vals) {
    super(BinOptimizedRandomnessPolicy.SampleBin);
    this.splits = splits.toArray(new FeatureBinarization.BinaryFeature[splits.size()]);
    this.masks = new boolean[masks.size()];
    for (int i = 0; i < this.masks.length; ++i) {
      this.masks[i] = masks.get(i);
    }
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

  public RandomnessAwareRegion(final List<FeatureBinarization.BinaryFeature> splits,
                               final List<Boolean> masks,
                               final RandomVariable[] vals) {
    super(BinOptimizedRandomnessPolicy.SampleBin);
    this.splits = splits.toArray(new FeatureBinarization.BinaryFeature[splits.size()]);
    this.masks = new boolean[masks.size()];
    for (int i = 0; i < this.masks.length; ++i) {
      this.masks[i] = masks.get(i);
    }
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
          int idx = 0;
          for (int depth = 0; depth < splits.length; ++depth) {
            final double value = policy == BinOptimizedRandomnessPolicy.SampleBin ? samplers[depth].instance(random, doc) : features[depth].expectation(doc);
            if (splits[depth].value(value) != masks[depth]) {
              break;
            }
            else {
              ++idx;
            }
          }
          final double instance = sampler[idx].instance(random);
          result.set(doc, instance);
        });
        return result;
      }
      case BinsExpectation: {
        IntStream.range(0, result.rows()).parallel().forEach(doc -> {
          double value = 0;

          double probContinue = 1.0;

          for (int depth = 0; depth < splits.length; ++depth) {
            final RandomVariable rv = features[depth].randomVariable(doc);
            if (splits[depth] instanceof FeatureBinarization.TakeEqualFeature) {
              probContinue *= splits[depth].value(features[depth].expectation(doc)) ? 1 : 0;
            }
            else {
              probContinue *= masks[depth] ? 1.0 - rv.cdf(splits[depth].border()) : rv.cdf(splits[depth].border());
            }
            value += (1.0 - probContinue) * values[depth].mean();
            if (Math.abs(probContinue) <= zeroProbThreshold) {
              break;
            }
          }
          if (Math.abs(probContinue) > zeroProbThreshold) {
            value += probContinue * values[splits.length].mean();
          }
          result.set(doc, value);
        });
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

  static final double zeroProbThreshold = 1e-5;


  public double value(final Vec feature) {
    final BinOptimizedRandomnessPolicy policy = activePolicy();
    final FastRandom random = random();

    switch (policy) {
      case PointEstimateBin:
      case SampleBin: {
        int idx = 0;
        for (int i = 0; i < splits.length; ++i) {
          RandomVariable rv = splits[i].owner().owner().compute(feature);
          final double value = policy == BinOptimizedRandomnessPolicy.SampleBin ? rv.sampler().instance(random) : rv.mean();
          if (splits[i].value(value) != masks[i]) {
            break;
          }
          else {
            ++idx;
          }
        }
        return sampler[idx].instance(random);
//        return sampler[index].instance(random);
      }
      case BinsExpectation: {

        double result = 0.0;
        double prob = 1.0;


        for (int i = 0; i < splits.length; ++i) {
          RandomVariable rv = splits[i].owner().owner().compute(feature);
          if (splits[i] instanceof FeatureBinarization.TakeEqualFeature) {
            prob *= splits[i].value(rv.mean()) ? 1 : 0;
          }
          else {
            prob *= masks[i] ? 1.0 - rv.cdf(splits[i].border()) : rv.cdf(splits[i].border());
          }
          result += (1.0 - prob) * values[i].mean();
          if (Math.abs(prob) <= zeroProbThreshold) {
            break;
          }
        }
        if (Math.abs(prob) > zeroProbThreshold) {
          result += prob * values[splits.length].mean();
        }
        return result;
      }
      default: {
        throw new RuntimeException("unimplemented");
      }
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
    if (!(o instanceof RandomnessAwareRegion)) return false;

    final RandomnessAwareRegion that = (RandomnessAwareRegion) o;

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
