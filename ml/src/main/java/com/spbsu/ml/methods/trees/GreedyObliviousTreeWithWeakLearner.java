package com.spbsu.ml.methods.trees;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.ml.BFGrid;
import com.spbsu.commons.math.Trans;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.tools.DataTools;
import com.spbsu.ml.func.Ensemble;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.StatBasedLoss;
import com.spbsu.ml.loss.WeightedLoss;
import com.spbsu.ml.methods.VecOptimization;
import com.spbsu.ml.methods.linearRegressionExperiments.WeakLeastAngle;
import com.spbsu.ml.models.ObliviousTree;

import java.util.List;
import java.util.Set;
import java.util.TreeSet;

import static com.spbsu.commons.math.vectors.VecTools.copy;

/**
 * User: noxoomo
 */

public class GreedyObliviousTreeWithWeakLearner<Loss extends StatBasedLoss> extends VecOptimization.Stub<Loss> {
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

    final List<BFGrid.BinaryFeature> conditions = ((ObliviousTree)result[0]).features();
    //damn java 7 without unique, filters, etc and autoboxing overhead…
    Set<Integer> uniqueFeatures = new TreeSet<>();
    for (BFGrid.BinaryFeature bf : conditions) {
      if (!bf.row().empty()
        )
        uniqueFeatures.add(bf.findex);
    }
//    //prototype
    while (uniqueFeatures.size() < 10) {
      final int feature = rand.nextInt(ds.data().columns());
      if (!base.grid.row(feature).empty())
        uniqueFeatures.add(feature);
    }

    Vec newTarget = copy(loss.target());
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

    Ensemble ensemble = new Ensemble(result, weights);
    return ensemble;
}

}

