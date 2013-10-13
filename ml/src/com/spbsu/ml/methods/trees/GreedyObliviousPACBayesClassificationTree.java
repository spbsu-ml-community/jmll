package com.spbsu.ml.methods.trees;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.Model;
import com.spbsu.ml.Oracle1;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.data.impl.BinarizedDataSet;
import com.spbsu.ml.data.impl.Bootstrap;
import com.spbsu.ml.loss.SigmoidClassificationError;
import com.spbsu.ml.methods.MLMethodOrder1;
import com.spbsu.ml.models.ObliviousTree;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static java.lang.Math.*;

/**
 * User: solar
 * Date: 30.11.12
 * Time: 17:01
 */
public class GreedyObliviousPACBayesClassificationTree implements MLMethodOrder1 {
  private final Random rng;
  private final BinarizedDataSet ds;
  private final int depth;
  private BFGrid grid;
  private Vec prevPoint;

  public GreedyObliviousPACBayesClassificationTree(Random rng, DataSet ds, BFGrid grid, int depth) {
    this.rng = rng;
    this.depth = depth;
    this.ds = new BinarizedDataSet(ds, grid);
    this.grid = grid;
    prevPoint = new ArrayVec(ds.power());
  }

  @Override
  public Model fit(DataSet learn, Oracle1 loss) {
    return fit(learn, loss, new ArrayVec(learn.power()));
  }

  @Override
  public ObliviousTree fit(DataSet ds, Oracle1 loss, Vec point) {
    if(!(loss instanceof SigmoidClassificationError))
      throw new IllegalArgumentException("Log likelihood with sigmoid value function supported only, found " + loss);
    final List<BFGrid.BinaryFeature> conditions = new ArrayList<BFGrid.BinaryFeature>(depth);
    final Vec target = ds instanceof Bootstrap ? ((Bootstrap)ds).original().target() : ds.target();
    PACBayesClassificationLeaf seed = new PACBayesClassificationLeaf(this.ds, point, target, VecTools.fill(new ArrayVec(point.dim()), 1.));
    PACBayesClassificationLeaf[] currentLayer = new PACBayesClassificationLeaf[]{seed};
    PACBayesClassificationLeaf seedPrev = new PACBayesClassificationLeaf(this.ds, prevPoint, target, VecTools.fill(new ArrayVec(point.dim()), 1.));
    PACBayesClassificationLeaf[] currentLayerPrev = new PACBayesClassificationLeaf[]{seedPrev};
    for (int level = 0; level < depth; level++) {
      double[] scores = new double[grid.size()];
      double[] s = new double[grid.size()];
      double[] sPrev = new double[grid.size()];
      for (int i = 0; i < currentLayer.length; i++) {
        Arrays.fill(s, 0);
        Arrays.fill(sPrev, 0);
        currentLayer[i].score(s);
        currentLayerPrev[i].score(sPrev);
        double sumPrev = 0;
        double minPrev = sPrev[ArrayTools.min(sPrev)];
        double maxPrev = sPrev[ArrayTools.max(sPrev)];
        for (int j = 0; j < sPrev.length; j++) {
          final double normalized = -(sPrev[j] - minPrev)*3/(maxPrev - minPrev);
          sumPrev += exp(normalized * normalized);
        }

        for (int j = 0; j < sPrev.length; j++) {
          if (currentLayer[i].size() > 0) {
            final double normalized = -(sPrev[j] - minPrev)*3/(maxPrev - minPrev + 0.1);
            final double p = exp(-normalized * normalized);
            final double R = sqrt((-log(p) + log(1 / 0.05)) / 2. / currentLayer[i].size());
            if(Double.isNaN(R))
              System.out.println();
            scores[j] += s[j] + (Double.isNaN(R) ? 0 : R);
          }
          else
            scores[j] += s[j];
        }
      }

      final int min = ArrayTools.min(scores);
      if (min < 0)
        throw new RuntimeException("Can not find optimal split!");
      BFGrid.BinaryFeature bestSplit = grid.bf(min);
      final PACBayesClassificationLeaf[] nextLayer = new PACBayesClassificationLeaf[1 << (level + 1)];
      final PACBayesClassificationLeaf[] nextLayerPrev = new PACBayesClassificationLeaf[1 << (level + 1)];
      double errors = 0;
      double realErrors = 0;
      for (int l = 0; l < currentLayer.length; l++) {
        PACBayesClassificationLeaf leaf = currentLayer[l];
        PACBayesClassificationLeaf leafPrev = currentLayerPrev[l];
        nextLayer[2 * l] = leaf;
        nextLayer[2 * l + 1] = leaf.split(bestSplit);
        nextLayerPrev[2 * l] = leafPrev;
        nextLayerPrev[2 * l + 1] = leafPrev.split(bestSplit);
        errors += nextLayer[2 * l].total.score();
        errors += nextLayer[2 * l + 1].total.score();
        realErrors += checkLeaf(point, target, nextLayer[2 * l]);
        realErrors += checkLeaf(point, target, nextLayer[2 * l + 1] );
      }
//      System.out.println("Estimated error: " + realErrors + " found: " + errors);
      conditions.add(bestSplit);
      currentLayer = nextLayer;
      currentLayerPrev = nextLayerPrev;
    }
    final PACBayesClassificationLeaf[] leaves = currentLayer;

    double[] values = new double[leaves.length];
    double[] weights = new double[leaves.length];
    {
      for (int i = 0; i < weights.length; i++) {
        values[i] = leaves[i].alpha();
        weights[i] = leaves[i].size();
      }
    }
    VecTools.assign(prevPoint, point);
    return new ObliviousTree(conditions, values, weights);
  }

  private double checkLeaf(Vec point, Vec target, PACBayesClassificationLeaf pacBayesClassificationLeaf) {
    double inc = pacBayesClassificationLeaf.total.alpha();
    double xx = 0;
    for (int i = 0; i < pacBayesClassificationLeaf.indices.length; i++) {
      int index = pacBayesClassificationLeaf.indices[i];
      final double y = target.get(index) > 0 ? 1 : -1;
      xx += 1./(1 + Math.exp(y * (point.get(index) + inc)));
    }
    return xx;
  }
}
