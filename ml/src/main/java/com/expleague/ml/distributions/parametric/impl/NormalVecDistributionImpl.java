package com.expleague.ml.distributions.parametric.impl;

import com.expleague.ml.distributions.RandomVec;
import com.expleague.ml.distributions.RandomVecBuilder;
import com.expleague.ml.distributions.parametric.NormalDistribution;
import com.expleague.ml.distributions.parametric.NormalVecDistribution;
import gnu.trove.list.array.TFloatArrayList;

import static com.expleague.commons.math.MathTools.sqr;

/**
 * Created by noxoomo on 27/10/2017.
 */
public class NormalVecDistributionImpl extends RandomVec.CoordinateIndependentStub implements NormalVecDistribution {
  private final TFloatArrayList means;
  private final TFloatArrayList sd;

  public NormalVecDistributionImpl() {
    means = new TFloatArrayList();
    sd = new TFloatArrayList();
  }

  public double expectation(final int idx) {
    return means.get(idx);
  }

  public RandomVecBuilder<NormalDistribution> builder() {
    return new Builder();
  }

  @Override
  public int length() {
    return means.size();
  }


  private void add(final NormalDistribution distribution) {
    means.add((float) distribution.mu());
    sd.add((float) distribution.sd());
  }


  public NormalVecDistribution add(final int idx, final double scale, final NormalDistribution var) {
    final double newMu = mu(idx) + var.mu() * scale;
    final double newVar = sqr(sd(idx)) + sqr(scale * var.sd());
    means.set(idx, (float) newMu);
    sd.set(idx, (float) Math.sqrt(newVar));
    return this;
  }

  @Override
  public NormalVecDistribution add(final NormalVecDistribution other, double scale) {
    for (int i = 0; i < means.size(); ++i) {
      final double newMu = mu(i) + other.mu(i) * scale;
      final double newSd = Math.sqrt(sqr(sd(i)) + sqr(scale * other.sd(i)));
      means.set(i, (float) newMu);
      sd.set(i, (float) newSd);
    }
    return this;
  }

  public NormalVecDistribution scale(int idx, double scale) {
    if (scale != 1.0) {
      means.set(idx, (float) (means.get(idx) * scale));
      sd.set(idx, (float) (sd.get(idx) * scale));
    }
    return this;
  }


  @Override
  public NormalVecDistribution scale(double scale) {
    if (scale != 1.0) {
      for (int i = 0; i < means.size(); ++i) {
        means.set(i, (float) (means.get(i) * scale));
        sd.set(i, (float) (sd.get(i) * scale));
      }
    }
    return this;
  }

  @Override
  public NormalDistribution at(int idx) {
    return new CoordinateImpl(this, idx);
  }

  @Override
  public double mu(int idx) {
    return means.get(idx);
  }

  @Override
  public double sd(int idx) {
    return sd.get(idx);
  }

  public static class Builder implements RandomVecBuilder<NormalDistribution> {
    private final NormalVecDistributionImpl impl;

    public Builder() {
      impl = new NormalVecDistributionImpl();
    }


    @Override
    public RandomVecBuilder<NormalDistribution> add(final NormalDistribution distribution) {
      impl.means.add((float) distribution.mu());
      impl.sd.add((float) distribution.sd());
      return this;
    }

    @Override
    public NormalVecDistribution build() {
      return impl;
    }
  }


  protected static class CoordinateImpl extends CoordinateProjectionStub<NormalVecDistributionImpl> implements NormalDistribution {

    protected CoordinateImpl(final NormalVecDistributionImpl owner,
                             final int idx) {
      super(owner, idx);
    }


    @Override
    public NormalDistribution add(NormalDistribution other, double scale) {
      owner.add(idx, scale, other);
      return this;
    }

    @Override
    public NormalDistribution scale(double scale) {
      owner.scale(idx, scale);
      return this;
    }

    @Override
    public double mu() {
      return owner.mu(idx);
    }

    @Override
    public double sd() {
      return owner.sd(idx);
    }
  }
}
