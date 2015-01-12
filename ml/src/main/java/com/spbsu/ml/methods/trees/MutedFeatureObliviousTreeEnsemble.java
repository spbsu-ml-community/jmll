package com.spbsu.ml.methods.trees;

import org.jetbrains.annotations.NotNull;

import java.util.Arrays;


import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.ml.func.Ensemble;
import com.spbsu.ml.models.ObliviousTree;

/**
 * Created by vkokarev on 09.12.14.
 */
public class MutedFeatureObliviousTreeEnsemble extends Ensemble<ObliviousTree> {
  private MutedFeatureObliviousTreeEnsemble(final ObliviousTree[] models, final Vec weights) {
    super(models, weights);
  }

  public static MutedFeatureObliviousTreeEnsemble from(@NotNull final Ensemble<ObliviousTree> origin, @NotNull final int ... mutedFeaturesIndexes) throws ClassCastException {
    final ObliviousTree[] newModels = new ObliviousTree[origin.models.length];
    final Vec weights = new ArrayVec(origin.weights.length());
    int realIdx = 0;
    for (int i = 0; i < origin.models.length; i++) {
      final ObliviousTree model = ObliviousTree.removeFeatures(origin.models[i], mutedFeaturesIndexes);
      if (model != null) {
        newModels[realIdx] = model;
        weights.set(realIdx++, origin.weights.get(i));
      }
    }
    return new MutedFeatureObliviousTreeEnsemble(Arrays.copyOf(newModels, realIdx), new ArrayVec(weights.toArray(), 0, realIdx));
  }
}
