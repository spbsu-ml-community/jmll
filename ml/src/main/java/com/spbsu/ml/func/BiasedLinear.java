package com.spbsu.ml.func;

import com.spbsu.commons.math.vectors.Vec;

/**
 * User: noxoomo
 */

public class BiasedLinear extends Linear {
  public final double bias;

  public BiasedLinear(final double[] weights, double bias) {
    super(weights);
    this.bias = bias;
  }

  public BiasedLinear(final Vec weights, double bias) {
    super(weights);
    this.bias = bias;
  }


  @Override
  public double value(final Vec point) {
    return super.value(point) + bias;
  }

  @Override
  public boolean equals(final Object o) {
    if (this == o) return true;
    if (!(o instanceof BiasedLinear)) return false;

    final BiasedLinear that = (BiasedLinear) o;
    return that.bias == this.bias && super.equals(that);
  }

  @Override
  public int hashCode() {
    return weights.hashCode() + (int)(bias);
  }
}
