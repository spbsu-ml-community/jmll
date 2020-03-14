package com.expleague.ml.methods.trees;

import com.expleague.commons.math.MathTools;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.util.ArrayTools;
import com.expleague.commons.util.Pair;
import com.expleague.ml.BFGrid;
import com.expleague.ml.Binarize;
import com.expleague.ml.data.Aggregate;
import com.expleague.ml.data.impl.BinarizedDataSet;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.loss.AdditiveLoss;
import com.expleague.ml.loss.L2;
import com.expleague.ml.loss.LinearL2;
import com.expleague.ml.loss.WeightedLoss;
import com.expleague.ml.methods.VecOptimization;
import com.expleague.ml.models.ObliviousLinearTree;
import com.expleague.ml.models.ObliviousTree;
import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.list.array.TIntArrayList;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.ListIterator;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

/**
 * User: solar
 * Date: 30.11.12
 * Time: 17:01
 */
public class GreedyObliviousLinearTree<Loss extends AdditiveLoss> extends VecOptimization.Stub<Loss> {
  private final int depth;
  public final BFGrid grid;

  public GreedyObliviousLinearTree(final BFGrid grid, final int depth) {
    this.grid = grid;
    this.depth = depth;
  }

  @Override
  public ObliviousLinearTree fit(final VecDataSet ds, final Loss loss) {
    List<TIntArrayList> leaves = new ArrayList<>(1);
    final List<BFGrid.Feature> conditions = new ArrayList<>(depth);
    double currentScore = Double.POSITIVE_INFINITY;

    final BinarizedDataSet bds =  ds.cache().cache(Binarize.class, VecDataSet.class).binarize(grid);

    final double[] pweights = loss instanceof WeightedLoss ? ((WeightedLoss) loss).weights() : ArrayTools.fill(new double[ds.length()], 1);
    final double[] scores = new double[grid.size()];
    final Vec target = ((L2) (loss instanceof WeightedLoss ? ((WeightedLoss) loss).base() : loss)).target();
    leaves.add(new TIntArrayList(learnPoints(loss, ds)));
    final List<Aggregate> aggregates = new ArrayList<>();
    final List<Vec> weights = new ArrayList<>();
    final TDoubleArrayList based = new TDoubleArrayList();

    for (int level = 0; level < depth; level++) {
      Arrays.fill(scores, 0.);
      aggregates.clear();

      final LinearL2 localTarget = new LinearL2(ds, conditions.toArray(new BFGrid.Feature[conditions.size()]), target);
      leaves.stream().map(TIntArrayList::toArray).forEach(points -> {
        final Aggregate agg = new Aggregate(bds, localTarget.statsFactory());
        agg.append(points, pweights);
        agg.visit((bf, left, right) -> scores[bf.index()] += localTarget.score((LinearL2.Stat)left) + localTarget.score((LinearL2.Stat)right));
        aggregates.add(agg);
      });

      final int bestSplit = ArrayTools.min(scores);
      if (bestSplit < 0 || scores[bestSplit] >= currentScore - 1e-9)
        break;
      final BFGrid.Feature bestSplitBF = grid.bf(bestSplit);

      weights.clear();
      based.clear();

      conditions.add(bestSplitBF);
//      final int[] projection = conditions.stream()./*filter(bf -> bf.row().size() > 2).*/mapToInt(BFGrid.Feature::findex).sorted().distinct().toArray();

      final List<TIntArrayList> next = new ArrayList<>(leaves.size() * 2);
//      System.out.println();
      for (int l = 0; l < leaves.size(); l++) {
        final TIntArrayList points = leaves.get(l);
        final Aggregate agg = aggregates.get(l);
        final LinearL2.Stat leftStat = (LinearL2.Stat)agg.combinatorForFeature(bestSplit);
        final LinearL2.Stat rightStat = (LinearL2.Stat)agg.total(bestSplitBF.findex()).remove(leftStat);
        final Vec wHatLeft = localTarget.optimalWeights(leftStat);
        final Vec wHatRight = localTarget.optimalWeights(rightStat);
        weights.add(wHatLeft);
        weights.add(wHatRight);
        final TIntArrayList left = new TIntArrayList(points.size());
        final TIntArrayList right = new TIntArrayList(points.size());
        final double[] basedLR = new double[]{0., 0.};
//        final double[] ll = new double[]{0., 0.};
        final byte[] bins = bds.bins(bestSplitBF.findex());
        points.forEach(idx -> {
//          final Vec x = ds.data().row(idx);
//          final double y = target.get(idx);
          if (!bestSplitBF.value(bins[idx])) {
            left.add(idx);
            basedLR[0] += pweights[idx];
//            final double yHat = wHatLeft.get(0) + VecTools.multiply(wHatLeft.sub(1, projection.length), (Vec) x.sub(projection));
//            ll[0] += MathTools.sqr(y - yHat);
          }
          else {
            right.add(idx);
            basedLR[1] += pweights[idx];
//            final double yHat = wHatRight.get(0) + VecTools.multiply(wHatRight.sub(1, projection.length), (Vec) x.sub(projection));
//            ll[1] += MathTools.sqr(y - yHat);
          }
          return true;
        });
//        System.out.println(ll[0] + " vs " + localTarget.value(leftStat) + " \\hat{w} = " + wHatLeft);
//        System.out.println(ll[1] + " vs " + localTarget.value(rightStat) + " \\hat{w} = " + wHatRight);
        based.add(basedLR[0]);
        based.add(basedLR[1]);
        next.add(left);
        next.add(right);
      }
      leaves = next;
      currentScore = scores[bestSplit];
    }

    return new ObliviousLinearTree(conditions, weights.toArray(new Vec[weights.size()]), based.toArray());
  }

  private int[] learnPoints(Loss loss, VecDataSet ds) {
    if (loss instanceof WeightedLoss) {
      return ((WeightedLoss) loss).points();
    } else return ArrayTools.sequence(0, ds.length());
  }
}
