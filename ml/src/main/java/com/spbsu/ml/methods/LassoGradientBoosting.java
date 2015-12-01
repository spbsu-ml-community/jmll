package com.spbsu.ml.methods;

import com.spbsu.commons.func.impl.WeakListenerHolderImpl;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.commons.math.Trans;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.tools.DataTools;
import com.spbsu.ml.func.Ensemble;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.L2Reg;

import java.util.ArrayList;
import java.util.List;


/**
 * Created by noxoomo on 09/02/15.
 */

public class LassoGradientBoosting<GlobalLoss extends L2> extends WeakListenerHolderImpl<LassoGradientBoosting.LassoGBIterationResult> implements VecOptimization<GlobalLoss> {
  protected final VecOptimization<L2> weak;
  private final Class<? extends L2> factory;
  int iterationsCount;
  double lambda = 1e-3;
  double alpha = 1.0;

  public  void setAlpha(double alpha) {
    this.alpha = alpha;
  }

  public void setLambda(double lambda) {
    this.lambda = lambda;
  }

  public LassoGradientBoosting(final VecOptimization<L2> weak, final int iterationsCount) {
    this(weak, L2Reg.class, iterationsCount);
  }

  public LassoGradientBoosting(final VecOptimization<L2> weak, final Class<? extends L2> factory, final int iterationsCount) {
    this.weak = weak;
    this.factory = factory;
    this.iterationsCount = iterationsCount;
  }

  @Override
  public Ensemble fit(final VecDataSet learn, final GlobalLoss globalLoss) {
    Mx transformedData = new VecBasedMx(learn.data().rows(), iterationsCount);
    ElasticNetMethod.ElasticNetCache lassoCache = new ElasticNetMethod.ElasticNetCache(transformedData, globalLoss.target, 0, alpha, lambda);
    ElasticNetMethod lasso = new ElasticNetMethod(1e-5, alpha, lambda);

    final Vec cursor = new ArrayVec(globalLoss.xdim());
    final List<Trans> weakModels = new ArrayList<>(iterationsCount);
    final Vec weights = new ArrayVec(iterationsCount);
    final Trans gradient = globalLoss.gradient();

    for (int t = 0; t < iterationsCount; t++) {
      final Vec gradientValueAtCursor = gradient.trans(cursor);
      final L2 localLoss = DataTools.newTarget(factory, gradientValueAtCursor, learn);
      final Trans weakModel = weak.fit(learn, localLoss);
      weakModels.add(weakModel);
      Vec applied = weakModel.transAll(learn.data()).col(0);
      for (int row = 0; row < learn.data().rows(); ++row) {
        transformedData.set(row, t, -applied.get(row));
      }
      lassoCache.updateDim(t + 1);
      Vec currentWeights = lasso.fit(lassoCache).weights;
      {
        VecTools.fill(cursor, 0.0);
        for (int observation = 0; observation < cursor.dim(); ++observation) {
          for (int weakFeature = 0; weakFeature < weakModels.size(); ++weakFeature)
            cursor.adjust(observation, currentWeights.get(weakFeature) * transformedData.get(observation, weakFeature));
        }
      }
      invoke(new LassoGBIterationResult(weakModel, cursor, currentWeights));
    }
    return new Ensemble(ArrayTools.toArray(weakModels), weights);
  }

  public static class LassoGBIterationResult {
    public final Trans addedModel;
    public final Vec newWeights;
    public final Vec cursor;

    public LassoGBIterationResult(final Trans model, final Vec cursor, final Vec newWeights) {
      this.addedModel = model;
      this.cursor = cursor;
      this.newWeights = newWeights;
    }
  }
}



