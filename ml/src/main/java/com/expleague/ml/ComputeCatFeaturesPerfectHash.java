package com.expleague.ml;

import com.expleague.commons.func.Computable;
import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.ml.data.perfectHash.PerfectHash;
import com.expleague.ml.data.perfectHash.impl.FeaturePerfectHash;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.randomnessAware.ProcessRandomnessPolicy;
import com.expleague.ml.randomnessAware.VecRandomFeatureExtractor;

import java.util.HashMap;
import java.util.Map;
import java.util.WeakHashMap;

/**
 * noxoomo
 */
public class ComputeCatFeaturesPerfectHash implements Computable<VecDataSet, ComputeCatFeaturesPerfectHash> {
  private VecDataSet dataSet;
  private Map<Integer, PerfectHash<Vec>> hash  = new HashMap<>();

  public PerfectHash<Vec> hash(final int feature) {
    if (!hash.containsKey(feature)) {
      final FeaturePerfectHash perfectHash = new FeaturePerfectHash(feature);
      final Mx data = dataSet.data();
      for (int i = 0; i < dataSet.length(); ++i) {
        perfectHash.add(data.row(i));
      }

      synchronized(this) {
        if (!hash.containsKey(feature)) {
          hash.put(feature, perfectHash);
        }
      }
    }
    final PerfectHash<Vec> result;
    synchronized (this) {
      result = hash.get(feature);
    }
    return result;
  }

  @Override
  public ComputeCatFeaturesPerfectHash compute(final VecDataSet dataSet) {
    this.dataSet = dataSet;
    return this;
  }
}
