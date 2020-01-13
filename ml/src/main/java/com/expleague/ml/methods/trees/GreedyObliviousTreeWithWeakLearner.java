package com.expleague.ml.methods.trees;

import com.expleague.commons.math.Trans;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.random.FastRandom;
import com.expleague.ml.BFGrid;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.data.tools.DataTools;
import com.expleague.ml.func.Ensemble;
import com.expleague.ml.loss.L2;
import com.expleague.ml.loss.AdditiveLoss;
import com.expleague.ml.loss.WeightedLoss;
import com.expleague.ml.methods.VecOptimization;
import com.expleague.ml.methods.linearRegressionExperiments.WeakLeastAngle;
import com.expleague.ml.models.ObliviousTree;

import java.util.List;
import java.util.Set;
import java.util.TreeSet;

/**
 * User: noxoomo
 */

public class GreedyObliviousTreeWithWeakLearner<Loss extends L2> extends VecOptimization.Stub<Loss> {
  private final GreedyObliviousTree<WeightedLoss<Loss>> base;
  private final FastRandom rand;

  public GreedyObliviousTreeWithWeakLearner(
    final GreedyObliviousTree<WeightedLoss<Loss>> base,
    final FastRandom rand) {
    this.base = base;
    this.rand = rand;
  }

  private int[] learnPoints(WeightedLoss loss) {
    return loss.points();
  }

  @Override
  public Trans fit(final VecDataSet ds, final Loss loss) {
    final WeightedLoss<Loss> bsLoss = DataTools.bootstrap(loss, rand);
    final Trans[] result = new Trans[2];
    result[0] = base.fit(ds, bsLoss);

    final List<BFGrid.Feature> conditions = ((ObliviousTree)result[0]).features();
    //damn java 7 without unique, filters, etc and autoboxing overhead…
    Set<Integer> uniqueFeatures = new TreeSet<>();
    for (BFGrid.Feature bf : conditions) {
      if (!bf.row().empty()
        )
        uniqueFeatures.add(bf.findex());
    }
//    //prototype
    while (uniqueFeatures.size() < 10) {
      final int feature = rand.nextInt(ds.data().columns());
      if (!base.grid.row(feature).empty())
        uniqueFeatures.add(feature);
    }

    Vec newTarget = VecTools.copy(loss.target());
    Vec predictions = result[0].transAll(ds.data()).col(0);
    for (int i = 0; i < predictions.dim(); ++i)
      newTarget.adjust(i, -predictions.get(i));

    final int[] features = new int[uniqueFeatures.size()];
    {
      int j = 0;
      for (Integer i : uniqueFeatures) {
        features[j++] = i;
      }
    }

    L2 localLoss = DataTools.newTarget(L2.class,newTarget,ds);

    WeakLeastAngle regression = new WeakLeastAngle(learnPoints(bsLoss), features);
    result[1] = regression.fit(ds,localLoss);

    Vec weights = new ArrayVec(2);
    VecTools.fill(weights,1.0);

    return new Ensemble(result, weights);
}

}

