package com.expleague.ml.methods;

import com.expleague.commons.func.impl.WeakListenerHolderImpl;
import com.expleague.commons.math.Trans;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.random.FastRandom;
import com.expleague.ml.TargetFunc;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.data.tools.DataTools;
import com.expleague.ml.distributions.RandomVariable;
import com.expleague.ml.distributions.RandomVec;
import com.expleague.ml.func.RandomFuncEnsemble;
import com.expleague.ml.loss.L2;
import com.expleague.ml.loss.L2Reg;
import com.expleague.ml.randomnessAware.RandomFunc;
import com.expleague.ml.randomnessAware.ProcessRandomnessPolicy;

import java.util.ArrayList;
import java.util.List;


/**
 * User: noxoomo
 */
public class RandomnessAwareGradientBoosting<GlobalLoss extends TargetFunc> extends WeakListenerHolderImpl<RandomFunc> implements RandomnessAwareVecOptimization<GlobalLoss> {
  protected final RandomnessAwareVecOptimization<L2> weak;
  private final Class<? extends L2> factory;
  private Config config;
  private final FastRandom random = new FastRandom();

  public RandomnessAwareGradientBoosting(final RandomnessAwareVecOptimization<L2> weak,
                                         final Config config) {
    this(weak, L2Reg.class, config);
  }

  public RandomnessAwareGradientBoosting(final RandomnessAwareVecOptimization<L2> weak,
                                         final Class<? extends L2> factory,
                                         final Config config) {
    this.weak = weak;
    this.factory = factory;
    this.config = config;
  }

  @Override
  public RandomVec emptyVec(int dim) {
    return weak.emptyVec(dim);
  }

  @Override
  public RandomVariable emptyVar() {
    return weak.emptyVar();
  }

  @Override
  public RandomFuncEnsemble fit(final VecDataSet learn,
                                final GlobalLoss globalLoss) {

    final RandomVec cursor = weak.emptyVec(globalLoss.xdim());
    final List<RandomFunc> weakModels = new ArrayList<>(config.iterationsCount);
    final Trans gradient = globalLoss.gradient();
    final Vec gradientValueAtCursor = new ArrayVec(globalLoss.xdim());

    for (int t = 0; t < config.iterationsCount; t++) {
      gradient.transTo(cursor.instance(random), gradientValueAtCursor);
//      gradient.transTo(cursor.expectation(), gradientValueAtCursor);
      final L2 localLoss = DataTools.newTarget(factory, gradientValueAtCursor, learn);
      final RandomFunc weakModel = weak.fit(learn, localLoss);
      weakModels.add(weakModel);
      weakModel.appendTo(-config.step, learn, cursor);
      invoke(new RandomFuncEnsemble<>(weakModels, -config.step));
    }
    return new RandomFuncEnsemble<>(weakModels, -config.step);
  }


  public static class Config {
    int iterationsCount;
    double step;

    public Config(final int iterationsCount,
                  final double step) {
      this.iterationsCount = iterationsCount;
      this.step = step;
    }
  }
}
