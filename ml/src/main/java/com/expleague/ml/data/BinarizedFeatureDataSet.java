package com.expleague.ml.data;

import com.expleague.commons.func.CacheHolder;
import com.expleague.commons.func.ScopedCache;
import com.expleague.commons.random.FastRandom;
import com.expleague.ml.ComputeBinarization;
import com.expleague.ml.ComputeBinarizedFeature;
import com.expleague.ml.ComputeRowOffsets;
import com.expleague.ml.FeatureBinarization;
import com.expleague.ml.data.impl.BinarizedFeature;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.models.BinOptimizedRandomnessPolicy;
import com.expleague.ml.randomnessAware.VecRandomFeatureExtractor;

import java.util.*;

/**
 * User: noxoomo
 */

public class BinarizedFeatureDataSet implements CacheHolder {
  private final ScopedCache cache = new ScopedCache(getClass(), this);
  private final VecDataSet dataSet;
  private final List<BinarizedFeature> features;
  private final GridHelper gridHelper;

  public BinarizedFeatureDataSet(final VecDataSet dataSet,
                                 final List<BinarizedFeature> features) {
    this.dataSet = dataSet;
    this.features = features;
    this.gridHelper = new GridHelper();

  }

  public VecDataSet owner() {
    return dataSet;
  }

  public List<BinarizedFeature> features() {
    return features;
  }

  public BinarizedFeature binarizedFeature(final VecRandomFeatureExtractor featureExtractor) {
    return dataSet.cache().cache(ComputeBinarizedFeature.class, VecDataSet.class).get(featureExtractor);
  }

  public int size() {
    return features().size();
  }

  @Override
  public ScopedCache cache() {
    return cache;
  }

  public GridHelper gridHelper() {
    return gridHelper;
  }

  public static class Builder {
    private final VecDataSet basedOn;
    private final int binCount;
    private final FastRandom random;
    private final List<BinarizedFeature> features = new ArrayList<>();
    private BinOptimizedRandomnessPolicy policy;

    public Builder(final VecDataSet basedOn,
                   final int binCount,
                   final FastRandom random) {
      this.basedOn = basedOn;
      this.binCount = binCount;
      this.random = random;
    }


    public Builder addFeature(final VecRandomFeatureExtractor extractor) {
      final FeatureBinarization featureBinarization = basedOn.cache().cache(ComputeBinarization.class, VecDataSet.class).computeBinarization(extractor, random, binCount);
      features.add(basedOn.cache().cache(ComputeBinarizedFeature.class, VecDataSet.class).build(extractor, featureBinarization, policy));
      return this;
    }

    public BinarizedFeatureDataSet build() {
      return new BinarizedFeatureDataSet(basedOn, features);
    }

    public void setPolicy(final BinOptimizedRandomnessPolicy policy) {
      this.policy = policy;
    }
  }

  public class GridHelper {
    private final Map<FeatureBinarization, Integer> featureIds;
    private final List<FeatureBinarization.BinaryFeature> binaryFeatures;

    GridHelper() {
      this.featureIds = new HashMap<>();
      binaryFeatures = new ArrayList<>();
      for (int i = 0; i < features.size(); ++i) {
        featureIds.put(features.get(i).binarization(), i);
        binaryFeatures.addAll(Arrays.asList(features.get(i).binarization().features()));
      }
    }

    private final int[] offsets = cache().cache(ComputeRowOffsets.class, BinarizedFeatureDataSet.class).offsets();

    public int[] binFeatureOffsets() {
      return offsets;
    }

    private final int[] binOffsets = cache().cache(ComputeRowOffsets.class, BinarizedFeatureDataSet.class).binOffsets();

    public int[] binOffsets() {
      return binOffsets;
    }

    public int binaryFeatureOffset(final FeatureBinarization.BinaryFeature binaryFeature) {
      final int[] offsets = binFeatureOffsets();
      return offsets[featureIds.get(binaryFeature.owner())] + binaryFeature.binId();
    }

    public int binFeatureCount() {
      final int[] offsets = binFeatureOffsets();
      return offsets[offsets.length - 1];
    }

    public int binCount() {
      final int[] offsets = binOffsets();
      return offsets[offsets.length - 1];
    }

    public FeatureBinarization.BinaryFeature binFeature(final int idx) {
      return binaryFeatures.get(idx);
    }
  }

}
