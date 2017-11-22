package com.expleague.ml.data.ctrs;

import com.expleague.commons.math.Func;
import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.distributions.RandomVec;
import com.expleague.ml.models.RandomVariableRandomnessPolicy;
import com.expleague.ml.randomnessAware.RandomnessAwareTrans;

public class CtrTrans extends RandomnessAwareTrans.Stub<RandomVariableRandomnessPolicy> implements RandomnessAwareTrans<RandomVariableRandomnessPolicy>, Func {
  private final Ctr<?> ctr;

  public CtrTrans(final RandomVariableRandomnessPolicy policy,
                  final Ctr<?> ctr) {
    super(policy);
    this.ctr = ctr;
  }


  @Override
  public double value(final Vec x) {
    switch (activePolicy()) {
      case Expectation: {
        return ctr.get(x).mean();
      }
      case Sample: {
        return ctr.get(x).sampler().instance(random());
      }
    }
    return 0;
  }

  @Override
  public int dim() {
    return ctr.dim();
  }


  @Override
  public Mx transAll(final VecDataSet dataSet) {
    final RandomVec<?> randomVec = ctr.apply(dataSet);
    switch (activePolicy()) {
      case Expectation: {
        return new VecBasedMx(1, randomVec.expectation());
      }
      case Sample: {
        return new VecBasedMx(1, randomVec.sampler().sample(random()));
      }
      default: {
        throw new RuntimeException("unknown policy type");
      }
    }
  }

  @Override
  public int xdim() {
    return dim();
  }

  @Override
  public int ydim() {
    return 1;
  }
}
