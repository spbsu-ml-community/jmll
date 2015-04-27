package com.spbsu.ml.methods.trees;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.WeightedLoss;
import com.spbsu.ml.models.ExponentialObliviousTree;
import com.spbsu.ml.models.ObliviousTree;

import java.util.List;

/*Created with IntelliJ IDEA.
    *User:towelenee
    *Date:30.11.13
    *Time:17:48
    *Idea please stop making my code yellow
*/

public class GreedyExponentialObliviousTree extends GreedyObliviousTree<WeightedLoss<L2>> {
  private final double DistCoef;

  public GreedyExponentialObliviousTree(BFGrid grid, int depth, double DistCoef) {
    super(grid, depth);
    this.DistCoef = DistCoef;
  }

  @Override
  public ObliviousTree fit(final VecDataSet ds, final WeightedLoss<L2> loss) {
    final ObliviousTree tree = super.fit(ds, loss);
    final List<BFGrid.BinaryFeature> features = tree.features();
    final int numberOfRegions = 1 << features.size();

    double[] targetSum = new double[numberOfRegions];
    double[] weightsSum = new double[numberOfRegions];
    for (int i = 0; i < ds.xdim(); ++i) {
      final double weight = loss.weight(i);
      final double target = loss.target().get(i);
      final Vec point = ds.data().row(i);

      for (int region = 0; region < numberOfRegions; ++i) {
        final double expWeight = ExponentialObliviousTree.getWeightOfPointInRegion(region, point, features, DistCoef);

        targetSum[region] += target * weight * expWeight;
        weightsSum[region] += weight * expWeight;
      }
    }
    double[] values = new double[numberOfRegions];
    for (int region = 0; region < numberOfRegions; ++region) {
      values[region] = targetSum[region] / weightsSum[region];
    }

    return new ObliviousTree(features, values, tree.based());
  }
}