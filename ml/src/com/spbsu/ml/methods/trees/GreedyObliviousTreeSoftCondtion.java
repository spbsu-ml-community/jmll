package com.spbsu.ml.methods.trees;

import com.spbsu.ml.BFGrid;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.WeightedLoss;
import com.spbsu.ml.methods.Optimization;
import com.spbsu.ml.models.ObliviousTree;

import java.util.List;

/**
 * Created by towelenee on 25.05.14.
 */
public class GreedyObliviousTreeSoftCondtion implements Optimization<WeightedLoss<L2>> {
  private final GreedyObliviousTree<WeightedLoss<L2>> got;
  private final double exponentialBase, probabilityLimit;
  private final int condtionSatisifate;
  private final int depth;
  private final BFGrid grid;

  public GreedyObliviousTreeSoftCondtion(BFGrid grid, int depth, double exp_base, double probability_limit, int condtionSatisifate) {
    this.exponentialBase = exp_base;
    this.probabilityLimit = probability_limit;
    this.condtionSatisifate = condtionSatisifate;
    got = new GreedyObliviousTree<WeightedLoss<L2>>(grid, depth);
    this.depth = depth;
    this.grid = grid;
  }

  public double getProbabilityBeingInRegion(byte[] bin, int region, List<BFGrid.BinaryFeature> features) {
    int depth = features.size();
    double ans = 1;
    for (int neighbourRegion = 0; neighbourRegion < 1 << depth; neighbourRegion++) {
      if (Integer.bitCount(region ^ neighbourRegion) <= depth - condtionSatisifate) {
        for (int i = 0; i < depth; i++)
          ans *= getProbabilitySatisfyCondtion(bin, features.get(i), (neighbourRegion >> i & 1) == 1);
      }
    }
    //System.out.println(ans);
    return ans;
  }

  private double getProbabilitySatisfyCondtion(byte[] bin, BFGrid.BinaryFeature binaryFeature, boolean greater) {
    if ((binaryFeature.value(bin)) != greater)
      return Math.pow(exponentialBase, Math.abs(bin[binaryFeature.findex] - binaryFeature.binNo));
    else
      return 1;
  }

  @Override
  public ObliviousTree fit(DataSet learn, WeightedLoss<L2> loss) {
    ObliviousTree base = got.fit(learn, loss);
    final List<BFGrid.BinaryFeature> features = base.features();
    double newValues[] = new double[1 << depth];
    int count[] = new int[1 << depth];

    for (int i = 0; i < learn.power(); i++) {
      byte[] bin = new byte[learn.xdim()];
      grid.binarize(learn.data().row(i), bin);
      int cnt = 0;
      for (int index = 0; index < (1 << depth); index++) {
        if (getProbabilityBeingInRegion(bin, index, features) >= probabilityLimit) {
          newValues[index] += loss.getWeights()[i] * loss.getMetric().target.get(i);
          count[index] += loss.getWeights()[i];
          //System.out.println(index);
          cnt++;
        }
      }
      if (cnt > 1) {
        for (int index = 0; index < (1 << depth); index++) {
          if (getProbabilityBeingInRegion(bin, index, features) >= probabilityLimit) {
            System.out.println(index);
            System.out.println(getProbabilityBeingInRegion(bin, index, features));

          }
        }
        System.exit(0);
      }
    }
    for (int i = 0; i < 1 << depth; i++)
      if (count[i] != 0) {
        newValues[i] /= count[i];
        //System.out.println(newValues[i]);
      }
    return new ObliviousTree(features, newValues, base.based());
  }
}
