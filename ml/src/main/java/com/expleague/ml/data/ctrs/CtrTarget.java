package com.expleague.ml.data.ctrs;

import com.expleague.commons.func.Factory;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.vectors.VecBuilder;
import com.expleague.commons.seq.Seq;
import com.expleague.ml.ComputeCatFeaturesPerfectHash;
import com.expleague.ml.GridTools;
import com.expleague.ml.TargetFunc;
import com.expleague.ml.bayesianEstimation.ConjugateBayesianEstimator;
import com.expleague.ml.bayesianEstimation.impl.BetaConjugateBayesianEstimator;
import com.expleague.ml.data.perfectHash.PerfectHash;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.distributions.DynamicRandomVec;
import com.expleague.ml.distributions.RandomVariable;
import com.expleague.ml.distributions.parametric.BetaDistribution;
import com.expleague.ml.distributions.parametric.impl.BetaVecDistributionImpl;
import com.expleague.ml.loss.L2;
import com.expleague.ml.loss.LLLogit;
import gnu.trove.list.array.TIntArrayList;

import java.util.Arrays;
import java.util.function.IntFunction;

public class CtrTarget {
  private Vec target;
  private CtrTargetType type;

  public CtrTarget(final Vec target,
                   final CtrTargetType type) {
    if (type == CtrTargetType.Binomial) {
      double[] values = target.toArray();
      Arrays.sort(values);
      final TIntArrayList borders = GridTools.greedyLogSumBorders(values, 1);
      final double border = values[borders.get(0) - 1];
      VecBuilder builder = new VecBuilder(target.dim());
      for (int i = 0; i < target.dim(); ++i) {
        builder.append(target.get(i) > border ? 1.0 : 0.0);
      }
      this.target = builder.build();
    } else {
      throw new RuntimeException("unimplemented");
    }
    this.type = type;
  }

  public Vec target() {
    return target;
  }

  public CtrTargetType type() {
    return type;
  }

  public enum CtrTargetType {
    Binomial,
    Normal
  }
}


//class CtrTargetHelper {
//
//  public static <GlobalLoss extends TargetFunc> IntFunction<Ctr<?>> catFeatureCtr(final VecDataSet ds,
//                                                                                  final GlobalLoss loss) {
//    final IntFunction<Ctr<?>> builder = (IntFunction<Ctr<?>>) featureId -> {
//      final PerfectHash<Vec> hash = ds.cache().cache(ComputeCatFeaturesPerfectHash.class, VecDataSet.class).hash(featureId);
//      if (loss instanceof LLLogit) {
//
//
//      }
//      else if (loss instanceof L2) {
//
//      }
//      else {
//        throw new RuntimeException("unsupported target for ctrs " + loss.toString());
//      }
//    };
//    return builder;
//  }

//  public <U extends RandomVariable<U>> ConjugateBayesianEstimator<?> createEstimator(Class<U> clazz) {
//    if (BernoulliDistribution.class.isAssignableFrom(clazz)) {
//      return new BetaConjugateBayesianEstimator();
//    } else {
//      throw new RuntimeException("Unknown random variable class");
//    }
//
//  }


//    public static <U extends RandomVariable<U>> Ctr<?> newCtr(final Class<U> model,
//    final Seq<?> values,
//    final PerfectHash<Vec> hash,
//    final U prior,
//    final VecDataSet ds) {
//      if (BetaDistribution.class.isAssignableFrom(model)) {
//        final DynamicRandomVec<U> betaVecDistribution = (DynamicRandomVec<U>) new BetaVecDistributionImpl();
//        final ConjugateBayesianEstimator<U> betaConjugateBayesianEstimator = (ConjugateBayesianEstimator<U>) new BetaConjugateBayesianEstimator();
//        return new Ctr<U>(betaVecDistribution, hash, prior, betaConjugateBayesianEstimator, ds.data().columns());
//      } else {
//        throw new RuntimeException("Error: unknown ctr model " + model.getSimpleName());
//      }
//}
//    newCtr(final Class<U> model,
//    final Seq<?> values,
//    final PerfectHash<Vec> hash,
//    final U prior,
//    final VecDataSet ds)
