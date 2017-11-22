package com.expleague.ml.methods;

import com.expleague.commons.func.impl.WeakListenerHolderImpl;
import com.expleague.commons.math.Trans;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.random.FastRandom;
import com.expleague.ml.TargetFunc;
import com.expleague.ml.data.ctrs.CtrTrans;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.data.tools.DataTools;
import com.expleague.ml.func.RandomnessAwareEnsemble;
import com.expleague.ml.loss.L2;
import com.expleague.ml.loss.L2Reg;
import com.expleague.ml.models.BinOptimizedRandomnessPolicy;
import com.expleague.ml.randomnessAware.RandomnessAwareTrans;
import com.expleague.ml.randomnessAware.ProcessRandomnessPolicy;

import java.util.ArrayList;
import java.util.List;


/**
 * User: noxoomo
 */
public class RandomnessAwareGradientBoosting<GlobalLoss extends TargetFunc> extends WeakListenerHolderImpl<RandomnessAwareTrans> implements RandomnessAwareVecOptimization<GlobalLoss> {
  protected final RandomnessAwareVecOptimization<L2> weak;
  private final Class<? extends L2> factory;
  private final ProcessRandomnessPolicy transPolicy;
  private Config config;
  private final FastRandom random = new FastRandom();

  public RandomnessAwareGradientBoosting(final RandomnessAwareVecOptimization<L2> weak,
                                         final Config config,
                                         final ProcessRandomnessPolicy transPolicy) {
    this(weak, L2Reg.class, transPolicy, config);
  }

  public RandomnessAwareGradientBoosting(final RandomnessAwareVecOptimization<L2> weak,
                                         final Class<? extends L2> factory,
                                         final ProcessRandomnessPolicy transPolicy,
                                         final Config config) {
    this.weak = weak;
    this.factory = factory;
    this.transPolicy = transPolicy;
    this.config = config;
  }

  @Override
  public RandomnessAwareEnsemble fit(final VecDataSet learn,
                                     final GlobalLoss globalLoss) {

    final Vec cursor = new ArrayVec(globalLoss.xdim());
    final List<RandomnessAwareTrans<ProcessRandomnessPolicy>> weakModels = new ArrayList<>(config.iterationsCount);
    final Trans gradient = globalLoss.gradient();
    final Vec gradientValueAtCursor = new ArrayVec(globalLoss.xdim());

    for (int t = 0; t < config.iterationsCount; t++) {
      gradient.transTo(cursor, gradientValueAtCursor);
      final L2 localLoss = DataTools.newTarget(factory, gradientValueAtCursor, learn);
      final RandomnessAwareTrans weakModel = weak.fit(learn, localLoss);
      weakModel.setRandom(random);
      if (!(weakModel instanceof CtrTrans)) {
        weakModel.changePolicy(transPolicy);
      }
      weakModels.add(weakModel);
      final Vec modelValues = weakModel.transAll(learn);
      VecTools.append(cursor, VecTools.scale(modelValues, -config.step));
      if  (!(weakModel instanceof CtrTrans)) {
        weakModel.changePolicy(config.scoreApplyPolicy);
      }

      invoke(new RandomnessAwareEnsemble<>(weakModels, -config.step, random));
    }
    return new RandomnessAwareEnsemble<>(weakModels, -config.step, random);
  }


  public static class Config {
    int iterationsCount;
    double step;
    ProcessRandomnessPolicy scoreApplyPolicy;

    public Config(final int iterationsCount,
                  final double step, final ProcessRandomnessPolicy policy) {
      this.iterationsCount = iterationsCount;
      this.step = step;
      this.scoreApplyPolicy = policy;
    }
  }
}
