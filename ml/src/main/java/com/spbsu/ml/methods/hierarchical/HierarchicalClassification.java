package com.spbsu.ml.methods.hierarchical;

import com.spbsu.commons.func.Computable;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.*;
import com.spbsu.ml.data.VectorizedRealTargetDataSet;
import com.spbsu.ml.data.tools.MCTools;
import com.spbsu.ml.data.impl.HierarchyTree;
import com.spbsu.ml.func.Ensemble;
import com.spbsu.ml.loss.multiclass.hier.HierLoss;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.MLLLogit;
import com.spbsu.ml.loss.SatL2;
import com.spbsu.ml.methods.GradientBoosting;
import com.spbsu.ml.methods.MultiClass;
import com.spbsu.ml.methods.VecOptimization;
import com.spbsu.ml.methods.trees.GreedyObliviousTree;
import com.spbsu.ml.models.HierarchicalModel;
import com.spbsu.ml.models.MultiClassModel;
import gnu.trove.list.TIntList;
import gnu.trove.list.array.TIntArrayList;

/**
 * User: qdeee
 * Date: 06.02.14
 */
public class HierarchicalClassification implements VecOptimization<HierLoss> {
  private int weakIters;
  private double weakStep;
  private BFGrid grid;

  public HierarchicalClassification(int weakIters, double weakStep) {
    this(weakIters, weakStep, null);
  }

  public HierarchicalClassification(final int weakIters, final double weakStep, final BFGrid grid) {
    this.weakIters = weakIters;
    this.weakStep = weakStep;
    this.grid = grid;
  }

  @Override
  public Trans fit(VectorizedRealTargetDataSet<?> learn, HierLoss hierLoss) {
    grid = GridTools.medianGrid(learn, 32);
    HierarchicalModel model = traverseFit(hierLoss.getHierRoot());
    return model;
  }

  private HierarchicalModel traverseFit(final HierarchyTree.Node node) {
    final VectorizedRealTargetDataSet ds = node.createDS();
    final TIntList labels = new TIntArrayList();
    final Func[] resultModels;

    if (ds != null) {
      final Vec normTarget = MCTools.normalizeTarget(ds.target(), labels);
      final MLLLogit globalLoss = new MLLLogit(normTarget);

      final GradientBoosting<MLLLogit> boosting = new GradientBoosting<MLLLogit>(new MultiClass(new GreedyObliviousTree<L2>(grid, 5), new Computable<Vec, L2>() {
        @Override
        public L2 compute(Vec argument) {
          return new SatL2(argument);
        }
      }), weakIters, weakStep);
      final ProgressHandler calcer = new ProgressHandler() {
        int index = 0;

        @Override
        public void invoke(Trans partial) {
          if ((index + 1) % 20 == 0) {
            double value = globalLoss.value(partial.transAll(ds.data()));
            System.out.println("Node#" + node.getCategoryId() + ", iter=" + index + ", MLLLogitValue=" + value);
          }
          index++;
        }
      };
      boosting.addListener(calcer);

      System.out.println("\n\nBoosting at node " + node.getCategoryId() + " is started, DS size=" + ds.length());
      final Ensemble ensemble = boosting.fit(ds, globalLoss);
      resultModels = MultiClassModel.joinBoostingResults(ensemble).dirs();

    }
    else {
      //this node has only one child, so we introduce max const func that will return this child with probability = 1
      resultModels = new Func[] {new Func.Stub() {
        @Override
        public double value(Vec x) {
          return Double.MAX_VALUE;
        }

        @Override
        public int dim() {
          return 0;
        }
      }};
      labels.add(node.getChildren().get(0).getCategoryId());
      labels.add(node.getCategoryId());
    }
    final HierarchicalModel hierModel = new HierarchicalModel(resultModels, labels);
    for (HierarchyTree.Node child : node.getChildren()) {
      if (child.isLeaf())
        continue;
      HierarchicalModel childModel = traverseFit(child);
      hierModel.addChild(childModel, child.getCategoryId());
    }
    return hierModel;
  }
}
