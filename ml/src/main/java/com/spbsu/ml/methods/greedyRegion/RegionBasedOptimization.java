package com.spbsu.ml.methods.greedyRegion;

import com.spbsu.ml.TargetFunc;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.methods.VecOptimization;
import com.spbsu.ml.models.Region;

/**
 * Created by noxoomo on 09/02/15.
 */
public abstract class RegionBasedOptimization<Loss extends TargetFunc> extends VecOptimization.Stub<Loss> {
  @Override
  public abstract Region fit(VecDataSet learn, Loss loss);
}
