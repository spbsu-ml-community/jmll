package com.spbsu.ml.methods.greedyRegion;

import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.loss.StatBasedLoss;
import com.spbsu.ml.models.Region;

/**
 * Created by noxoomo on 09/02/15.
 */
public class BinaryRegion<Loss extends StatBasedLoss> extends RegionBasedOptimization<Loss> {
  RegionBasedOptimization<Loss> inner;
  public BinaryRegion(RegionBasedOptimization<Loss> inner) {
    this.inner = inner;
  }

  @Override
  public Region fit(VecDataSet learn, Loss loss) {
    return new Region(inner.fit(learn,loss),1.0,0);
  }
}
