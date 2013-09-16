package com.spbsu.ml.methods;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.Model;
import com.spbsu.ml.Oracle1;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.data.Histogram;
import com.spbsu.ml.data.impl.BinarizedDataSet;
import com.spbsu.ml.data.impl.Bootstrap;
import com.spbsu.ml.loss.L2Loss;
import com.spbsu.ml.models.Region;
import gnu.trove.TIntArrayList;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * User: solar
 * Date: 15.11.12
 * Time: 15:19
 */
public class GreedyTDRegion implements MLMethodOrder1 {
  protected final BFGrid grid;
  protected final BinarizedDataSet bds;
  private double alpha = 0.3;
  private double betta = 0.1;

  public GreedyTDRegion(Random rng, DataSet ds, BFGrid grid) {
    this(rng, ds, grid, 0.3, 0.1);
  }

  public GreedyTDRegion(Random rng, DataSet ds, BFGrid grid, double alpha, double betta) {
    this.grid = grid;
    this.alpha = alpha;
    this.betta = betta;
    bds = new BinarizedDataSet(ds, grid);
  }

  @Override
  public Model fit(DataSet learn, Oracle1 loss) {
    return fit(learn, loss, new ArrayVec(learn.power()));
  }

  public Model fit(DataSet ds, Oracle1 loss, Vec start) {
    assert loss.getClass() == L2Loss.class;
    DataSet learn = ds;
    final List<Region.BinaryCond> conditions = new ArrayList<Region.BinaryCond>(grid.size());
    final Vec target = ds instanceof Bootstrap ? ((Bootstrap) ds).original().target() : learn.target();
    int[] indices = ds instanceof Bootstrap ? ((Bootstrap) ds).order() : ArrayTools.sequence(0, ds.power());
    double[] scores = new double[grid.size()];
    final boolean[] masks = new boolean[grid.size()];
    double total = VecTools.sum(learn.target());
    double totalWeight = indices.length;
    double currentScore = Double.MAX_VALUE;

    while(true) {
      final Histogram histogram = bds.buildHistogram(learn.target(), start, indices);
      final int complexity = conditions.size() + 1;
      final double ftotal = total;
      final double ftotalWeight = totalWeight;
      Arrays.fill(scores, 0.);

      histogram.score(scores, new Histogram.Judge() {
        int bf = 0;
        @Override
        public double score(double sum, double sum2, double weight) {
          double rightScore = GreedyTDRegion.this.score(weight, sum, complexity);
          double leftScore = GreedyTDRegion.this.score(ftotalWeight - weight, ftotal - sum, complexity);
          masks[bf] = rightScore < leftScore;
          return Math.min(rightScore, leftScore);
        }
      });
      int bestBF = ArrayTools.min(scores);
      if (scores[bestBF] > currentScore)
        break;
      final Region.BinaryCond bestSplit = new Region.BinaryCond();
      bestSplit.bf = grid.bf(bestBF);
      bestSplit.mask = masks[bestBF];

      TIntArrayList inducedIndices = new TIntArrayList(indices.length);
      byte[] bins = bds.bins(bestSplit.bf.findex);
      double totalReduce = 0;
      for (int t = 0; t < indices.length; t++) {
        final int index = indices[t];
        if ((bins[index] > bestSplit.bf.binNo) == bestSplit.mask)
          inducedIndices.add(index);
        else
          totalReduce += target.get(index);
      }
      if (inducedIndices.isEmpty() || inducedIndices.size() == indices.length)
        break;
      total -= totalReduce;
      conditions.add(bestSplit);
      {
        boolean[] mask = new boolean[learn.power()];
        for (int t = 0; t < inducedIndices.size(); t++) {
          mask[inducedIndices.get(t)] = true;
        }
        final Region region = new Region(conditions, total / indices.length, indices.length, currentScore);
        for (int t = 0; t < indices.length; t++) {
          final int index = indices[t];
          final Vec point = learn.data().row(index);
          if (region.contains(point) != mask[index])
            System.out.println();
        }
      }

      indices = inducedIndices.toNativeArray();
      totalWeight = indices.length;
      currentScore = scores[bestBF];
    }

    return new Region(conditions, total/indices.length, indices.length, currentScore);
  }

  public double score(double count, double sum, int complexity) {
    if (count <= 0)
      return 0;
    final double err = -sum * sum / count;
    return err * (1. - alpha - alpha * Math.log(2)/ Math.log(count + 1.)) * (1 - betta - betta * complexity/(1. + complexity));
  }
}
