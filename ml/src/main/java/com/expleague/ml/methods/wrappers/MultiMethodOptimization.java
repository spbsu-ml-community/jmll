package com.expleague.ml.methods.wrappers;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.random.FastRandom;
import com.expleague.commons.math.Func;
import com.expleague.commons.math.Trans;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.loss.AdditiveLoss;
import com.expleague.ml.methods.VecOptimization;

public class MultiMethodOptimization<Loss extends AdditiveLoss> extends VecOptimization.Stub<Loss>  {
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
