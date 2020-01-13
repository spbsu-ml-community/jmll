package com.expleague.ml.methods.greedyRegion;

import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.loss.AdditiveLoss;
import com.expleague.ml.models.Region;

/**
 * Created by noxoomo on 09/02/15.
 */
public class BinaryRegion<Loss extends AdditiveLoss> extends RegionBasedOptimization<Loss> {
  RegionBasedOptimization<Loss> inner;
  public BinaryRegion(RegionBasedOptimization<Loss> inner) {
    this.inner = inner;
  }

  @Override
  public Region fit(VecDataSet learn, Loss loss) {
    return new Region(inner.fit(learn,loss),1.0,0);
  }
}
