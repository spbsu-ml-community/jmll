package com.expleague.ml.distributions;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.random.FastRandom;
import com.expleague.commons.seq.Seq;


public interface RandomVec extends RandomSeq<Double> {

  RandomVariable at(final int idx);

  double instance(final int idx, final FastRandom random);

  double logDensity(final int idx, final double value);

  double cdf(final int idx, final double value);

  Vec instance(final FastRandom random);


  public abstract class CoordinateIndependentStub implements RandomVec{

    public double logProb(final Seq<Double> data) {
      double ll = 0;
      if (data instanceof Vec) {
        for (int i = 0; i < length(); ++i) {
          ll += logDensity(i, ((Vec) data).get(i));
        }
      } else {
        for (int i = 0; i < length(); ++i) {
          ll += logDensity(i, data.at(i));
        }
      }
      return ll;
    };

    @Override
    public Vec instance(final FastRandom random) {
      Vec result = new ArrayVec(length());
      for (int i = 0; i < result.dim(); ++i) {
        result.set(i, instance(i, random));
      }
      return result;
    }
  }

  public abstract class CoordinateProjectionStub<Owner extends RandomVec>  implements RandomVariable {
    protected final Owner owner;
    protected final int idx;

    public CoordinateProjectionStub(final Owner owner, int idx) {
      this.idx = idx;
      this.owner = owner;
    }


    @Override
    public double logDensity(final double value) {
      return owner.logDensity(idx, value);
    }

    @Override
    public double cdf(final double value) {
      return owner.cdf(idx, value);
    }

    @Override
    public double sample(FastRandom random) {
      return owner.instance(idx, random);
    }
  }

  default Vec expectation() {
    Vec vec = new ArrayVec(length());
    for (int i = 0; i < vec.dim(); ++i) {
      vec.set(i, at(i).expectation());
    }
    return vec;
  }
}

