package com.expleague.ml.randomnessAware;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.distributions.DistributionConvolution;
import com.expleague.ml.distributions.RandomVariable;
import com.expleague.ml.distributions.RandomVec;

/**
 * Created by noxoomo on 22/10/2017.
 */
public interface RandomFunc  {

  RandomVariable emptyVar();

  RandomVec emptyVec(int dim);


  RandomVariable appendTo(final double scale,
                          final Vec vec,
                          final RandomVariable to);


  RandomVec appendTo(final double scale,
                     final VecDataSet dataSet,
                     final RandomVec dst);


  default RandomVariable appendTo(final Vec vec,
                          final RandomVariable to) {
    return appendTo(1.0, vec, to);
  }


  default RandomVec appendTo(final VecDataSet dataSet,
                             final RandomVec dst) {
    return appendTo(1.0, dataSet, dst);

  }

  int dim();

  public abstract class Stub implements RandomFunc {
    protected final DistributionConvolution convolution;

    public Stub(DistributionConvolution convolution) {
      this.convolution = convolution;
    }

    @Override
    public RandomVariable emptyVar() {
      return convolution.empty();
    }

    @Override
    public RandomVec emptyVec(int dim) {
      return convolution.empty(dim);
    }
  }

}




