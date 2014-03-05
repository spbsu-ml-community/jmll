package com.spbsu.ml.methods;

import com.spbsu.commons.func.Action;
import com.spbsu.commons.func.Computable;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.ml.*;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.data.impl.HierDS;
import com.spbsu.ml.func.Ensemble;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.MLLLogit;
import com.spbsu.ml.loss.SatL2;
import com.spbsu.ml.methods.trees.GreedyObliviousTree;
import com.spbsu.ml.models.HierarchicalModel;
import com.spbsu.ml.models.MultiClassModel;

/**
 * User: qdeee
 * Date: 06.02.14
 */
public class HierarchicalClassification implements Optimization<L2>{
  private int weakIters;
  private double weakStep;

  public HierarchicalClassification(int weakIters, double weakStep) {
    this.weakIters = weakIters;
    this.weakStep = weakStep;
  }

  @Override
  public Trans fit(DataSet learn, L2 loss) {
    HierDS treeStructure = ((HierDS) learn);

    HierDS.CategoryNode node = treeStructure.getRoot();
    while (node.getChildren().size() != 0) {
      node = node.getChildren().get(0);
    }
    if (node.getInnerDS() != null)
      throw new IllegalArgumentException("Hierarchy was not pruned!");

    HierDS.traversePrint(treeStructure.getRoot());
    HierarchicalModel model = traverseFit2(treeStructure.getRoot());
    return model;
  }

  private class MaxConstFunc extends Func.Stub {
    @Override
    public double value(Vec x) {
      return Double.MAX_VALUE;
    }

    @Override
    public int dim() {
      return 0;
    }
  }

  private HierarchicalModel traverseFit2(HierDS.CategoryNode node) {
    DataSet ds = node.getInnerDS();
    Func[] resultModels;
    if (ds == null) {
      resultModels = new Func[1];
      resultModels[0] = new MaxConstFunc();
    }
    else {
      BFGrid grid = GridTools.medianGrid(ds, 32);
      final int catId = node.getCategoryId();
      GradientBoosting<MLLLogit> boosting = new GradientBoosting<MLLLogit>(new MultiClass(new GreedyObliviousTree<L2>(grid, 5), new Computable<Vec, L2>() {
        @Override
        public L2 compute(Vec argument) {
          return new SatL2(argument);
        }
      }), weakIters, weakStep);
      final Action counter = new ProgressHandler() {
        int index = 0;

        @Override
        public void invoke(Trans partial) {
          System.out.println("Node#" + catId+ ", iter=" + index++);
        }
      };
      boosting.addListener(counter);
      System.out.println("Boosting at node " + node.getCategoryId() + " is started, DS size=" + node.getInnerDS().power());
      Ensemble<MultiClassModel> ensemble = (Ensemble<MultiClassModel>) boosting.fit(ds, new MLLLogit(ds.target()));
      Trans[] mcModels = ensemble.models;

      int classesCount = ensemble.ydim();
      Func[][] allModels = new Func[classesCount][ensemble.size()];
      for (int c = 0; c < allModels.length; c++) {
        for (int iter = 0; iter < allModels[c].length; iter++) {
          allModels[c][iter] = ((MultiClassModel)mcModels[iter]).dirs()[c];
        }
      }
      resultModels = new Func[classesCount];
      for (int i = 0; i < resultModels.length; i++) {
        resultModels[i] = new FuncEnsemble(allModels[i], VecTools.fill(new ArrayVec(ensemble.size()), weakStep));
      }
    }
    HierarchicalModel hierModel = new HierarchicalModel(resultModels, node.getNonEmptyLabels());
    for (HierDS.CategoryNode child : node.getChildren()) {
      if (child.isLeaf())
        continue;
      HierarchicalModel childModel = traverseFit2(child);
      hierModel.addChildren(childModel, child.getCategoryId());
    }
    return hierModel;
  }

  private class FuncEnsemble extends Ensemble<Func> implements Func{

    public FuncEnsemble(Func[] models, Vec weights) {
      super(models, weights);
    }

    public double value(Vec x) {
      double result = 0.;
      for (int i = 0; i < size(); i++) {
        result += models[i].value(x) * weights.get(i);
      }
      return result;
    }

    @Override
    public int dim() {
      return super.xdim();
    }
  }
}
