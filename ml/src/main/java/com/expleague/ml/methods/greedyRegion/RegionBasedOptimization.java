package com.expleague.ml.methods.greedyRegion;

import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.methods.VecOptimization;
import com.expleague.ml.TargetFunc;
import com.expleague.ml.models.Region;

/**
 * Created by noxoomo on 09/02/15.
 */
public abstract class RegionBasedOptimization<Loss extends TargetFunc> extends VecOptimization.Stub<Loss> {
  @Override
  public abstract Region fit(VecDataSet learn, Loss loss);
}
