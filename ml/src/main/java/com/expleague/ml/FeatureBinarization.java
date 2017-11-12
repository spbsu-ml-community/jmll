package com.expleague.ml;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.random.FastRandom;
import com.expleague.ml.distributions.RandomVec;
import com.expleague.ml.randomnessAware.VecRandomFeatureExtractor;
import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.list.array.TIntArrayList;

import java.util.Arrays;

/**
 * Created by noxoomo on 27/10/2017.
 */
public class FeatureBinarization {
  private final double[] borders;
  private final BinaryFeature[] features;
  private final Type type;
  private final VecRandomFeatureExtractor owner;

  public Type type() {
    return type;
  }

  public double[] borders() {
    return borders;
  }

  public BinaryFeature[] features() {
    return features;
  }

  public int bin(double value) {
    int index = 0;
    while (index < borders.length && value > borders[index])
      index++;
    return index;
  }

  public VecRandomFeatureExtractor owner() {
    return owner;
  }

  private FeatureBinarization(final double[] borders,
                              final Type type,
                              final VecRandomFeatureExtractor owner) {
    this.borders = borders;
    this.owner = owner;
    features = new BinaryFeature[this.borders.length];
    this.type = type;
  }

  public interface BinaryFeature {
    int binId();

    boolean value(final double val);

    boolean value(final int binId);

    FeatureBinarization owner();

    float border();

  }

  public static class TakeEqualFeature implements BinaryFeature {
    private final int idx;
    private final FeatureBinarization owner;

    TakeEqualFeature(final int idx, final FeatureBinarization owner) {
      this.idx = idx;
      this.owner = owner;
    }


    @Override
    public String toString() {
      return owner().owner.toString() + " == " + Double.toString(owner.borders[idx]);
    }

    @Override
    public int binId() {
      return idx;
    }

    @Override
    public boolean value(final double val) {
      return val == owner.borders[idx];
    }

    @Override
    public boolean value(final int binId) {
      return binId == idx;
    }

    @Override
    public FeatureBinarization owner() {
      return owner;
    }

    @Override
    public float border() {
      throw new RuntimeException("unimplemented");
    }
  }

  public static class TakeGreaterFeature implements BinaryFeature {
    private final int idx;
    private final FeatureBinarization owner;

    TakeGreaterFeature(final int idx,
                       final FeatureBinarization owner) {
      this.idx = idx;
      this.owner = owner;
    }

    @Override
    public int binId() {
      return idx;
    }

    @Override
    public boolean value(final double val) {
      return val > owner.borders[idx];
    }

    @Override
    public boolean value(final int binId) {
      return binId > idx;
    }

    @Override
    public FeatureBinarization owner() {
      return owner;
    }

    @Override
    public float border() {
      return (float) owner.borders[idx];
    }

    @Override
    public String toString() {
      return owner().owner.toString() + " >" + Double.toString(owner.borders[idx]);
    }
  }

  public static class FeatureBinarizationBuilder {
    private boolean isFloatSplit = true;
    private int binFactor = 15;
    private TDoubleArrayList values = new TDoubleArrayList();
    private boolean quantileBinarization = false;

    public FeatureBinarizationBuilder buildOneHot(final boolean flag) {
      this.isFloatSplit = !flag;
      return this;
    }

    public FeatureBinarizationBuilder setBinFactor(final int binFactor) {
      this.binFactor = binFactor;
      return this;
    }

    public FeatureBinarizationBuilder addSample(final Vec sample) {
      for (int i = 0; i < sample.dim(); ++i) {
        values.add(sample.get(i));
      }
      return this;
    }

    public FeatureBinarizationBuilder useQuantileBinarization(final boolean flag) {
      this.quantileBinarization = flag;
      return this;
    }

    public FeatureBinarizationBuilder addSample(final RandomVec<?> vec,
                                                final FastRandom fastRandom) {
//      final Vec sample = vec.expectation();//vec.sampler().sample(fastRandom);
      final Vec sample = vec.sampler().sample(fastRandom);
      addSample(sample);
      return this;
    }

    FeatureBinarization build(final VecRandomFeatureExtractor owner) {
      final double borders[];

      double[] valuesArr = values.toArray();
      Arrays.sort(valuesArr);
      if (isFloatSplit && (binFactor < values.size())) {
        if (quantileBinarization) {
          borders = GridTools.quantileBorders(valuesArr, binFactor);
        }
        else {
          final TIntArrayList borderIds = GridTools.greedyLogSumBorders(valuesArr, binFactor);
          borders = new double[borderIds.size() - 1];
          for (int i = 0; i < borderIds.size() - 1; ++i) {
            borders[i] = valuesArr[borderIds.get(i) - 1];
          }
        }
      }
      else {
        borders = GridTools.sortUnique(new ArrayVec(valuesArr));
      }

      final FeatureBinarization result = new FeatureBinarization(borders, isFloatSplit ? Type.FloatSplit : Type.OneHot, owner);

      for (int i = 0; i < result.features.length; ++i) {
        if (isFloatSplit) {
          result.features[i] = new TakeGreaterFeature(i, result);
        }
        else {
          result.features[i] = new TakeEqualFeature(i, result);
        }
      }
      return result;
    }
  }

  public enum Type {
    OneHot,
    FloatSplit
  }
}
