package com.spbsu.ml.methods;

import com.spbsu.commons.func.Computable;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.ml.*;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.data.impl.ChangedTarget;
import com.spbsu.ml.data.impl.DataSetImpl;
import com.spbsu.ml.data.impl.Hierarchy;
import com.spbsu.ml.func.Ensemble;
import com.spbsu.ml.func.FuncEnsemble;
import com.spbsu.ml.loss.multiclass.hier.HierLoss;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.MLLLogit;
import com.spbsu.ml.loss.SatL2;
import com.spbsu.ml.methods.trees.GreedyObliviousTree;
import com.spbsu.ml.models.HierarchicalModel;
import com.spbsu.ml.models.MultiClassModel;
import gnu.trove.list.TIntList;
import gnu.trove.list.array.TIntArrayList;

/**
 * User: qdeee
 * Date: 06.02.14
 */
public class HierarchicalClassification implements Optimization<HierLoss>{
  private int weakIters;
  private double weakStep;
  private BFGrid grid;

  public HierarchicalClassification(int weakIters, double weakStep) {
    this.weakIters = weakIters;
    this.weakStep = weakStep;
  }

  @Override
  public Trans fit(DataSet learn, HierLoss hierLoss) {
    grid = GridTools.medianGrid(learn, 32);
    HierarchicalModel model = traverseFit(hierLoss.getHierRoot());
    return model;
  }

  private HierarchicalModel traverseFit(final Hierarchy.CategoryNode node) {
    final DataSet ds = node.getInnerDS();
    final TIntList labels = new TIntArrayList();
    final Func[] resultModels;

    if (ds != null) {
      final Vec normTarget = node.normalizeTarget(labels);
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

      System.out.println("\n\nBoosting at node " + node.getCategoryId() + " is started, DS size=" + ds.power());
      final Ensemble ensemble = boosting.fit(ds, globalLoss);
      resultModels = joinBoostingResults(ensemble).dirs();

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
    for (Hierarchy.CategoryNode child : node.getChildren()) {
      if (child.isLeaf())
        continue;
      HierarchicalModel childModel = traverseFit(child);
      hierModel.addChildren(childModel, child.getCategoryId());
    }
    return hierModel;
  }

  public static MultiClassModel joinBoostingResults(Ensemble ensemble){
    if (ensemble.last() instanceof MultiClassModel) {
      Func[] joinedModels;
      Func[][] transpose = new Func[ensemble.ydim()][ensemble.size()];
      for (int c = 0; c < transpose.length; c++) {
        for (int iter = 0; iter < transpose[c].length; iter++) {
          final MultiClassModel mcm = (MultiClassModel) ensemble.models[iter];
          transpose[c][iter] = mcm.dirs()[c];
        }
      }
      joinedModels = new Func[ensemble.ydim()];
      for (int i = 0; i < joinedModels.length; i++) {
        joinedModels[i] = new FuncEnsemble(transpose[i], ensemble.weights);
      }
      return new MultiClassModel(joinedModels);
    }
    else
      throw new ClassCastException("Ensemble object does not contain MultiClassModel objects");
  }
}
