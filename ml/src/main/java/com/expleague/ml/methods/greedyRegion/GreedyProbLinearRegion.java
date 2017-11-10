package com.expleague.ml.methods.greedyRegion;

import com.expleague.commons.math.FuncC1;
import com.expleague.commons.math.Trans;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.random.FastRandom;
import com.expleague.ml.BFGrid;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.loss.StatBasedLoss;
import com.expleague.ml.methods.VecOptimization;

/**
 * User: solar
 * Date: 15.11.12
 * Time: 15:19
 */
public class GreedyProbLinearRegion<Loss extends StatBasedLoss> extends VecOptimization.Stub<Loss> {
  protected final BFGrid grid;
  private final FastRandom rng;

  public GreedyProbLinearRegion(final BFGrid grid, FastRandom rng) {
    this.grid = grid;
    this.rng = rng;
  }

  @Override
  public Trans fit(VecDataSet learn, Loss loss) {
    return null;
  }

  public static class LinearRegion extends FuncC1.Stub {
    @Override
    public double value(Vec x) {
      return 0;
    }

    @Override
    public int dim() {
      return 0;
    }
  }
}
