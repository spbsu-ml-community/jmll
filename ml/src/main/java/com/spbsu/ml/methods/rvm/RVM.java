package com.spbsu.ml.methods.rvm;

import com.spbsu.commons.random.FastRandom;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.func.BiasedLinear;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.methods.VecOptimization;

/**
 * Created by noxoomo on 03/06/15.
 */
public class RVM extends VecOptimization.Stub<L2> {
  private double tolerance = 1e-3;
  private FastRandom random;

  public RVM(FastRandom random) {
    this.random = new FastRandom(random.nextLong());
  }


  @Override
  public BiasedLinear fit(VecDataSet learn, L2 l2) {
    final RVMCache cache = new RVMCache(learn.data(),l2.target,random);
    return cache.fit(tolerance);
  }
}
