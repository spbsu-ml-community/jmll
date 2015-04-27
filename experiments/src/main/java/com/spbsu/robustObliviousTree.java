package com.spbsu;

import com.spbsu.ml.BFGrid;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.WeightedLoss;
import com.spbsu.ml.methods.trees.GreedyObliviousTree;
import com.spbsu.ml.models.ObliviousTree;

/**
 * Created by towelenee on 4/1/15.
 */
public class robustObliviousTree extends GreedyObliviousTree<WeightedLoss<L2>> {
  public robustObliviousTree(BFGrid grid, int depth) {
    super(grid, depth);
  }

  @Override
  public ObliviousTree fit(final VecDataSet ds, final WeightedLoss<L2> loss) {
    ObliviousTree tree = super.fit(ds, loss);
    double sum1[] = new double[tree.values().length];
    double sum2[] = new double[tree.values().length];
    double mean[] = new double[tree.values().length];
    double dispersion2[] = new double[tree.values().length];
    double scale[] = new double[tree.values().length];
    for (int i = 0; i < ds.xdim(); ++i) {
      int bin = tree.bin(ds.at(i));
      double target = loss.target().get(i);
      double weight = loss.weight(i);
      sum1[bin] += target * weight;
      sum2[bin] += target * target * weight;
    }

    double[] values = new double[tree.values().length];

    for (int i = 0; i < tree.values().length; ++i) {
      if (tree.based()[i] > 1) {
        double n = tree.based()[i];
        mean[i] = sum1[i] / n;
        dispersion2[i] = (sum2[i] - (sum1[i] * sum1[i]) / n) / (n - 1);
        if (dispersion2[i] > 1e-9) {
          scale[i] = 1. / dispersion2[i] / Math.sqrt(2 * Math.PI);
          dispersion2[i] *= dispersion2[i];
        }
      }
      if (dispersion2[i] < 1e-9) {
        values[i] = tree.values()[i];
      }
    }

    for (int i = 0; i < ds.xdim(); ++i) {
      int bin = tree.bin(ds.at(i));
      if (dispersion2[i] > 1e-9) {
        double target = loss.target().get(i);
        double shift = target - mean[bin];
        double weight = Math.exp(-shift * shift / 2 / dispersion2[bin]) * scale[bin];
        values[bin] += weight * target * loss.weight(i) / tree.based()[bin];
      }
    }
    return new ObliviousTree(tree.features(), values, tree.based());
  }
}