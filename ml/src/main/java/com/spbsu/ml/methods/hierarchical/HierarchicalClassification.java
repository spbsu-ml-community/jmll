package com.spbsu.ml.methods.hierarchical;

import com.spbsu.commons.seq.IntSeq;
import com.spbsu.ml.*;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.loss.blockwise.BlockwiseMLLLogit;
import com.spbsu.ml.methods.VecOptimization;
import gnu.trove.map.TIntObjectMap;
import gnu.trove.set.TIntSet;
import gnu.trove.set.hash.TIntHashSet;
import gnu.trove.stack.TIntStack;
import gnu.trove.stack.array.TIntArrayStack;

import java.util.Arrays;

/**
 * User: qdeee
 * Date: 06.02.14
 */
public class HierarchicalClassification extends VecOptimization.Stub<BlockwiseMLLLogit> {
  protected final VecOptimization<BlockwiseMLLLogit> weak;
  protected final TIntObjectMap<int[]> deps;
  protected final int root;

  public HierarchicalClassification(final VecOptimization<BlockwiseMLLLogit> weak, final TIntObjectMap<int[]> deps, final int root) {
    this.weak = weak;
    this.deps = deps;
    this.root = root;
  }

  @Override
  public Trans fit(final VecDataSet learn, final BlockwiseMLLLogit globalLoss) {
    final int[] weights = new int[learn.length()];
    final int[] localClasses = new int[learn.length()];
    final BlockwiseMLLLogit localLoss = new BlockwiseMLLLogit(new IntSeq(localClasses), learn);
    final TIntStack toProcess = new TIntArrayStack();

    toProcess.push(root);
    while (toProcess.size() > 0) {
      final TIntSet uniqClasses = new TIntHashSet(2);
      final int currentId = toProcess.pop();
      for (int i = 0; i < learn.length(); i++) {
        final int idx = Arrays.binarySearch(deps.get(currentId), globalLoss.label(i));
        if (idx != -1) {
          weights[i] = 1;
          localClasses[i] = idx;
          uniqClasses.add(idx);
        }
        else {
          weights[i] = 0;
        }
      }

      final Trans model = weak.fit(learn, localLoss);

//      BlockwiseWeightedLoss<BlockwiseMLLLogit> weightedLoss = (BlockwiseWeightedLoss<BlockwiseMLLLogit>) new BlockwiseWeightedLoss<>(globalLoss, weights);
    }

    return null;
  }
//  private int weakIters;
//  private double weakStep;
//  private BFGrid grid;
//
//  public HierarchicalClassification(final int weakIters, final double weakStep, final BFGrid grid) {
//    this.weakIters = weakIters;
//    this.weakStep = weakStep;
//    this.grid = grid;
//  }
//
//  @Override
//  public Trans fit(VecDataSet learn, HierLoss hierLoss) {
//    grid = GridTools.medianGrid(learn, 32);
//    return traverseFit(learn, hierLoss, hierLoss.getHierRoot());
//  }
//
//  private HierarchicalModel traverseFit(final VecDataSet learn, final HierLoss loss, final HierarchyTree.Node node) {
//    final VecDataSet ds = node.createDS(learn);
//    final TIntList labels = new TIntArrayList();
//    final Func[] resultModels;
//
//    if (ds != null) {
//      final MLLLogit globalLoss = MCTools.normalizeTarget(loss.original(), labels);
//
//      final GradientBoosting<MLLLogit> boosting = new GradientBoosting<MLLLogit>(new MultiClass(new GreedyObliviousTree<L2>(grid, 5), new Computable<Vec, L2>() {
//        @Override
//        public L2 compute(Vec argument) {
//          return new SatL2(argument);
//        }
//      }), weakIters, weakStep);
//      final ProgressHandler calcer = new ProgressHandler() {
//        int index = 0;
//
//        @Override
//        public void invoke(Trans partial) {
//          if ((index + 1) % 20 == 0) {
//            double value = globalLoss.value(partial.transAll(DataTools.data(ds)));
//            System.out.println("Node#" + node.getCategoryId() + ", iter=" + index + ", MLLLogitValue=" + value);
//          }
//          index++;
//        }
//      };
//      boosting.addListener(calcer);
//
//      System.out.println("\n\nBoosting at node " + node.getCategoryId() + " is started, DS size=" + ds.length());
//      final Ensemble ensemble = boosting.fit(ds, globalLoss);
//      resultModels = MultiClassModel.joinBoostingResults(ensemble).dirs();
//
//    }
//    else {
//      //this node has only one child, so we introduce max const func that will return this child with probability = 1
//      resultModels = new Func[] {new Func.Stub() {
//        @Override
//        public double value(Vec x) {
//          return Double.MAX_VALUE;
//        }
//
//        @Override
//        public int dim() {
//          return 0;
//        }
//      }};
//      labels.add(node.getChildren().get(0).getCategoryId());
//      labels.add(node.getCategoryId());
//    }
//    final HierarchicalModel hierModel = new HierarchicalModel(resultModels, labels);
//    for (HierarchyTree.Node child : node.getChildren()) {
//      if (child.isLeaf())
//        continue;
//      HierarchicalModel childModel = traverseFit(learn, loss, child);
//      hierModel.addChild(childModel, child.getCategoryId());
//    }
//    return hierModel;
//  }
}
