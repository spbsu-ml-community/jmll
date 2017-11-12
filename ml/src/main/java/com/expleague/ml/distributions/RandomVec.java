package com.expleague.ml.distributions;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.ml.distributions.samplers.RandomVariableSampler;
import com.expleague.ml.distributions.samplers.RandomVecSampler;

/**
 * Created by noxoomo on 22/10/2017.
 */
public interface RandomVec<U extends RandomVariable<U>> extends Distribution<Vec> {

  U randomVariable(final int idx);

  RandomVecBuilder<U> builder();

  RandomVec<U> setRandomVariable(final int idx, final U var);

  //  default Vec logDensity(Vec point, Vec to) {
//    for (int i = 0; i < point.dim(); ++i) {
//      to.set(i, logDensity(i, point));
//    }
//    return to;
//  }
//
  RandomVecSampler sampler();

  Vec expectationTo(Vec to);

  default Vec expectation() {
    Vec result = new ArrayVec(dim());
    return expectationTo(result);
  }

  int dim();

  double expectation(final int idx);

  double cumulativeProbability(int idx, double x);


  public abstract class IndependentCoordinatesDistribution<U extends RandomVariable<U>> implements RandomVec<U> {

    @Override
    public Vec expectationTo(final Vec to) {
      for (int i = 0; i < dim(); ++i) {
        to.set(i, expectation(i));
      }
      return to;
    }

//    public double logLikelihood(final Vec object) {
//      double sum = 0;
//      for (int i = 0; i < dim(); ++i) {
//        sum += randomVariable(i).logLikelihood(object.get(i));
//      }
//      return sum;
//    }


//  protected abstract double logDensity(final int idx, final double x);

    protected abstract class CoordinateProjectionStub<D extends IndependentCoordinatesDistribution<U>> implements RandomVariable<U> {
      protected final D owner;
      protected final int idx;

      protected CoordinateProjectionStub(final D owner,
                                         final int idx) {
        this.owner = owner;
        this.idx = idx;
      }

      @Override
      public double cdf(final double value) {
        return cumulativeProbability(idx, value);
      }

      //    @Override
//    public double logDensity(final double x) {
//      return owner.logDensity(idx, x);
//    }
//
      @Override
      public double mean() {
        return owner.expectation(idx);
      }

      @Override
      public RandomVariableSampler sampler() {
        return random -> owner.sampler().instance(random, idx);
      }

    }
  }
}


