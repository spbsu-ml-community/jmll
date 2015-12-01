package com.spbsu.ml.methods.greedyRegion.cherry;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.Binarize;
import com.spbsu.commons.math.Func;
import com.spbsu.commons.math.Trans;
import com.spbsu.ml.data.cherry.CherryLoss;
import com.spbsu.ml.data.cherry.CherryPick;
import com.spbsu.ml.data.cherry.CherrySubset;
import com.spbsu.ml.data.impl.BinarizedDataSet;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.loss.StatBasedLoss;
import com.spbsu.ml.loss.WeightedLoss;
import com.spbsu.ml.methods.VecOptimization;
import com.spbsu.ml.models.CNF;

import java.util.ArrayList;
import java.util.List;

/**
 * User: solar
 * Date: 15.11.12
 * Time: 15:19
 */
public class GreedyTDCherryRegion<Loss extends StatBasedLoss> extends VecOptimization.Stub<Loss> {
  public final BFGrid grid;
  private final CherryPick pick = new CherryPick();
  public GreedyTDCherryRegion(final BFGrid grid) {
    this.grid = grid;
  }

  private int[] learnPoints(Loss loss, VecDataSet ds) {
    if (loss instanceof WeightedLoss) {
      return ((WeightedLoss) loss).points();
    } else return ArrayTools.sequence(0, ds.length());
  }

  @Override
  public CNF fit(final VecDataSet learn,  final Loss loss) {
    final List<CNF.Clause> conditions = new ArrayList<>(100);
    final BinarizedDataSet bds = learn.cache().cache(Binarize.class, VecDataSet.class).binarize(grid);
    int[] points = learnPoints(loss, learn);
    double currentScore = Double.NEGATIVE_INFINITY;
    CherryLoss localLoss;
    {
      localLoss = new OutLoss3<>(new CherrySubset(bds,loss.statsFactory(),points), loss);
//      RankedDataSet rds = learn.cache().cache(RankIt.class, VecDataSet.class).value();
//      localLoss = new OutLoss<>(new CherryStochasticSubset(rds, bds, loss.statsFactory(), points), loss);
    }

    double bestIncInside = 0;
    double bestIncOutside = 0;
    while (true) {
      final CNF.Clause clause = pick.fit(localLoss);
      final double score = localLoss.score();
      if (score <= currentScore + 1e-9) {
        break;
      }

      System.out.println("\nAdded clause " + clause);
      currentScore = score;
      bestIncInside = localLoss.insideIncrement();
      bestIncOutside = localLoss.outsideIncrement();
      conditions.add(clause);
    }
    return  new CNF(conditions.toArray(new CNF.Clause[conditions.size()]), bestIncInside, bestIncOutside, grid);
  }
}

class MultiMethodOptimization<Loss extends StatBasedLoss> extends VecOptimization.Stub<Loss>  {
  private final VecOptimization<Loss>[] learners;
  private final FastRandom random;

  public MultiMethodOptimization(VecOptimization<Loss>[] learners, FastRandom random) {
    this.learners = learners;
    this.random = random;
  }

  class FuncHolder extends Func.Stub {
    Func inside;
    FuncHolder(Func inside) {
      this.inside = inside;
    }

    @Override
    public double value(Vec x) {
      return inside.value(x);
    }

    @Override
    public int dim() {
      return inside.dim();
    }
  }

  @Override
  public Trans fit(VecDataSet learn, Loss loss) {
    return new FuncHolder((Func)learners[random.nextInt(learners.length)].fit(learn,loss));
  }

}

