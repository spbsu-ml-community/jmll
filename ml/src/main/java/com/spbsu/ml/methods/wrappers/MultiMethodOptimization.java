package com.spbsu.ml.methods.wrappers;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.math.Func;
import com.spbsu.commons.math.Trans;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.loss.StatBasedLoss;
import com.spbsu.ml.methods.VecOptimization;

public class MultiMethodOptimization<Loss extends StatBasedLoss> extends VecOptimization.Stub<Loss>  {
  private final VecOptimization<Loss>[] learners;
  private final FastRandom random;

  public MultiMethodOptimization(VecOptimization<Loss>[] learners, FastRandom random) {
    this.learners = learners;
    this.random = random;
  }

  class FuncHolder extends Func.Stub {
    Func inside;
    FuncHolder(Func inside) {
      this.inside = inside;
    }

    @Override
    public double value(Vec x) {
      return inside.value(x);
    }

    @Override
    public int dim() {
      return inside.dim();
    }
  }

  @Override
  public Trans fit(VecDataSet learn, Loss loss) {
    return new FuncHolder((Func)learners[random.nextInt(learners.length)].fit(learn,loss));
  }

}
