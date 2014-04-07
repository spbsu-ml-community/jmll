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
import com.spbsu.ml.loss.hier.HierLoss;
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
  private int minEntries;

  public HierarchicalClassification(int weakIters, double weakStep, int minEntries) {
    this.weakIters = weakIters;
    this.weakStep = weakStep;
    this.minEntries = minEntries;
  }

  public HierarchicalClassification(int weakIters, double weakStep) {
    this(weakIters, weakStep, 10);
  }

  @Override
  public Trans fit(DataSet learn, HierLoss hierLoss) {
    HierarchicalModel model = traverseFit(hierLoss.getHierRoot());
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

  private HierarchicalModel traverseFit(Hierarchy.CategoryNode node) {
    DataSetImpl ds = (DataSetImpl) node.getInnerDS();
    Func[] resultModels;
    TIntList labels;
    if (ds != null) {
      labels = new TIntArrayList();
      Vec target = node.normalizeTarget(labels);
      ds = new ChangedTarget(ds, target);
      BFGrid grid = GridTools.medianGrid(ds, 32);
      GradientBoosting<MLLLogit> boosting = new GradientBoosting<MLLLogit>(new MultiClass(new GreedyObliviousTree<L2>(grid, 5), new Computable<Vec, L2>() {
        @Override
        public L2 compute(Vec argument) {
          return new SatL2(argument);
        }
      }), weakIters, weakStep);

      final MLLLogit globalLoss = new MLLLogit(ds.target());

      final int catId = node.getCategoryId();

      final DataSet learn = ds;
      ProgressHandler calcer = new ProgressHandler() {
        int index = 0;

        @Override
        public void invoke(Trans partial) {
          if ((index + 1) % 20 == 0) {
            double value = globalLoss.value(partial.transAll(learn.data()));
            System.out.println("Node#" + catId + ", iter=" + index + ", MLLLogitValue=" + value);
          }
          index++;
        }
      };
      boosting.addListener(calcer);

      System.out.println("Boosting at node " + node.getCategoryId() + " is started, DS size=" + ds.power());
      Ensemble<MultiClassModel> ensemble = (Ensemble<MultiClassModel>) boosting.fit(ds, globalLoss);
      double MLLLogitValue = globalLoss.value(ensemble.transAll(learn.data()));
      System.out.println("MLLLogitValue = " + MLLLogitValue);
      System.out.println("\n\n\n");
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
    else {
      //this node has only one child, so we introduce max const func that will return this child with probability = 1
      resultModels = new Func[] {new MaxConstFunc()};
      labels = new TIntArrayList();
      labels.add(node.getChildren().get(0).getCategoryId());
      labels.add(node.getCategoryId());
    }
    HierarchicalModel hierModel = new HierarchicalModel(resultModels, labels);
    for (Hierarchy.CategoryNode child : node.getChildren()) {
      if (child.isLeaf())
        continue;
      HierarchicalModel childModel = traverseFit(child);
      hierModel.addChildren(childModel, child.getCategoryId());
    }
    return hierModel;
  }

  private static class FuncEnsemble extends Ensemble<Func> implements Func{

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
