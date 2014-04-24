package com.spbsu.ml.methods;

import com.spbsu.commons.func.Computable;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.ml.*;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.data.DataTools;
import com.spbsu.ml.data.impl.HierarchyTree;
import com.spbsu.ml.func.Ensemble;
import com.spbsu.ml.func.FuncEnsemble;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.LLLogit;
import com.spbsu.ml.loss.MLLLogit;
import com.spbsu.ml.loss.SatL2;
import com.spbsu.ml.loss.multiclass.hier.HierLoss;
import com.spbsu.ml.methods.trees.GreedyObliviousTree;
import com.spbsu.ml.models.HierJoinedBinClassModel;
import com.spbsu.ml.models.HierarchicalModel;
import com.spbsu.ml.models.MCModel;
import gnu.trove.list.TIntList;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.list.linked.TIntLinkedList;

import java.util.ArrayList;
import java.util.List;

/**
 * User: qdeee
 * Date: 10.04.14
 */
public class HierarchicalRefinedClassification implements Optimization<HierLoss> {
  private int weakIters;
  private double weakStep;
  private BFGrid grid;

  public HierarchicalRefinedClassification(final int weakIters, final double weakStep) {
    this.weakIters = weakIters;
    this.weakStep = weakStep;
  }

  @Override
  public Trans fit(final DataSet learn, final HierLoss hierLoss) {
    grid = GridTools.medianGrid(learn, 32);
    final HierJoinedBinClassModel hierJoinedBinClassModel = traverseFitBottomTop(hierLoss.getHierRoot());
    final HierarchicalModel hierarchicalModel = traverseFitTopBottom(hierLoss.getHierRoot(), hierJoinedBinClassModel);
    return hierarchicalModel;
  }

  private HierJoinedBinClassModel traverseFitBottomTop(final HierarchyTree.Node node) {
    List<HierJoinedBinClassModel> childrenModels = new ArrayList<HierJoinedBinClassModel>(node.getChildren().size());
    TIntList childrenModelsId = new TIntArrayList(node.getChildren().size());
    for (HierarchyTree.Node child : node.getChildren()) {
      if (!child.isLeaf()) {
        childrenModels.add(traverseFitBottomTop(child));
        childrenModelsId.add(child.getCategoryId());
      }
    }

    final HierJoinedBinClassModel hierModel;
    if (node.isTrainingNode()) {
      final TIntList labels = new TIntArrayList();
      for (HierarchyTree.Node child : node.getChildren()) {
        labels.add(child.getCategoryId());
      }
      if (node.hasOwnDS())
        labels.add(node.getCategoryId());

      System.out.println(String.format("\n\nBoosting at node %d is started, DS size=%d", node.getCategoryId(), node.getEntriesCount()));
      final Func[] resultModels = new Func[labels.size()];
      for (int c = 0; c < labels.size(); c++) {
        final DataSet childDS = node.createDSForChild(labels.get(c));
        final LLLogit globalLoss = new LLLogit(childDS.target());

        final int classIndex = labels.get(c);
        final GradientBoosting<LLLogit> boosting = new GradientBoosting<LLLogit>(new GreedyObliviousTree<L2>(grid, 5), weakIters, weakStep);
        final ProgressHandler calcer = new ProgressHandler() {
          int iter = 0;

          @Override
          public void invoke(Trans partial) {
            if ((iter + 1) % 20 == 0) {
              double value = globalLoss.value(partial.transAll(childDS.data()));
              System.out.print(String.format("Node#%d, class=%d, positive examples=%d, iter=%d, LLLogit=%s\r",
                  node.getCategoryId(), classIndex, DataTools.classEntriesCount(childDS.target(), 1), iter, value));
            }
            iter++;
          }
        };
        boosting.addListener(calcer);

        Ensemble ensemble = boosting.fit(childDS, globalLoss);
        System.out.println();
        resultModels[c] = new FuncEnsemble(ArrayTools.map(ensemble.models, Func.class, new Computable<Trans, Func>() {
          @Override
          public Func compute(final Trans argument) {
            return (Func)argument;
          }
        }), ensemble.weights);
      }

      hierModel = new HierJoinedBinClassModel(resultModels, labels);
    }
    else {
      //this node has only one child, so we introduce max const func that will return this child with probability = 1
      final Func[] resultModels = {
          new Func.Stub() {
            @Override
            public double value(Vec x) {
              return Double.MAX_VALUE;
            }
            @Override
            public int dim() {
              return 0;
            }
          }
      };
      final TIntArrayList classLabels = new TIntArrayList(new int[] {node.getChildren().get(0).getCategoryId()});
      hierModel = new HierJoinedBinClassModel(resultModels, classLabels);
    }
    for (int i = 0; i < childrenModels.size(); i++) {
      final HierJoinedBinClassModel child = childrenModels.get(i);
      final int catId = childrenModelsId.get(i);
      hierModel.addChildren(child, catId);
    }
    return hierModel;
  }

  private HierarchicalModel traverseFitTopBottom(final HierarchyTree.Node node, final HierJoinedBinClassModel btModel) {
    final TIntList labels = new TIntArrayList(2);
    final Func[] resultModels;

    if (node.isTrainingNode()) {
      final TIntList idxs = node.joinLists();
      final DataSet ds = node.createDS();
      final Vec predicted = btModel.bestClassAll(ds.data());
      final TIntList errors = new TIntLinkedList();
      for (int i = 0; i < predicted.dim(); i++) {
        if (ds.target().get(i) != predicted.get(i)) {
          node.removeIdx(idxs.get(i));
          errors.add(idxs.get(i));
        }
      }
      if (errors.size() > 0) {
        node.addErrorChild(errors);
      }

      final DataSet learnDS = node.createDS();

      final Vec normTarget = DataTools.normalizeTarget(learnDS.target(), labels);
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
            double value = globalLoss.value(partial.transAll(learnDS.data()));
            System.out.print("Node#" + node.getCategoryId() + ", iter=" + index + ", MLLLogitValue=" + value + "\r");
          }
          index++;
        }
      };
      boosting.addListener(calcer);

      System.out.println("\n\nBoosting at node " + node.getCategoryId() + " is started, DS size=" + learnDS.power());
      final Ensemble ensemble = boosting.fit(learnDS, globalLoss);
      System.out.println();
      resultModels = MCModel.joinBoostingResults(ensemble).dirs();
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
    HierarchicalModel hierarchicalModel = new HierarchicalModel(resultModels, labels);

    for (HierarchyTree.Node childNode : node.getChildren()) {
      if (childNode.isLeaf())
        continue;
      final HierJoinedBinClassModel childBottomTopModel = btModel.getModelByLabel(childNode.getCategoryId());
      assert childBottomTopModel != null;
      final HierarchicalModel childTopBottomModel = traverseFitTopBottom(childNode, childBottomTopModel);
      if (childTopBottomModel != null) {
        hierarchicalModel.addChildren(childTopBottomModel, childNode.getCategoryId());
      }
    }

    return hierarchicalModel;
  }

}
