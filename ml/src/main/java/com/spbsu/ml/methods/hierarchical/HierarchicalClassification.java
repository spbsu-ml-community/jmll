package com.spbsu.ml.methods.hierarchical;

import com.spbsu.commons.seq.IntSeq;
import com.spbsu.commons.util.tree.FastTree;
import com.spbsu.commons.util.tree.Node;
import com.spbsu.commons.util.tree.NodeVisitor;
import com.spbsu.commons.util.tree.Tree;
import com.spbsu.commons.util.tree.impl.node.InternalNode;
import com.spbsu.commons.util.tree.impl.node.LeafNode;
import com.spbsu.ml.*;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.loss.blockwise.BlockwiseMLLLogit;
import com.spbsu.ml.loss.blockwise.BlockwiseWeightedLoss;
import com.spbsu.ml.methods.GradientBoosting;
import com.spbsu.ml.methods.VecOptimization;
import com.spbsu.ml.models.HierarchicalModel;
import com.spbsu.ml.models.MultiClassModel;
import gnu.trove.list.TIntList;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.list.linked.TIntLinkedList;
import gnu.trove.set.TIntSet;
import gnu.trove.set.hash.TIntHashSet;

import java.util.Arrays;
import java.util.List;
import java.util.Stack;

/**
 * User: qdeee
 * Date: 06.02.14
 */
public class HierarchicalClassification extends VecOptimization.Stub<BlockwiseMLLLogit> {
  protected final VecOptimization<TargetFunc> weak;
  protected final FastTree tree;

  public HierarchicalClassification(final VecOptimization<TargetFunc> weak, final FastTree tree) {
    this.weak = weak;
    this.tree = tree;
  }

  @Override
  public Trans fit(final VecDataSet learn, final BlockwiseMLLLogit globalLoss) {
    //avoiding a lot of allocations
    final int[] weights = new int[learn.length()];
    final int[] localClasses = new int[learn.length()];

    final NodeVisitor<HierarchicalModel> learner = new NodeVisitor<HierarchicalModel>() {
      @Override
      public HierarchicalModel visit(final InternalNode node) {
        final TIntList uniqClasses = new TIntLinkedList();
        for (Node child : node.getChildren()) {
          uniqClasses.add(child.id);
        }

        for (int i = 0; i < learn.length(); i++) {
          final int dsClassLabel = globalLoss.label(i);
          final List<Node> children = node.getChildren();
          weights[i] = 0;
          localClasses[i] = -1;
          for (int j = 0; j < children.size(); j++) {
            final Node child = children.get(j);
            if (tree.isFirstDescendantOfSecondOrEqual(dsClassLabel, child.id)) {
              weights[i] = 1;
              localClasses[i] = j;
              break;
            }
          }
        }

        final BlockwiseWeightedLoss<BlockwiseMLLLogit> localWeightedLoss = new BlockwiseWeightedLoss<>(
            new BlockwiseMLLLogit(new IntSeq(localClasses), learn),
            weights
        );
        final MultiClassModel model = (MultiClassModel) weak.fit(learn, localWeightedLoss);
        final HierarchicalModel hierarchicalModel = new HierarchicalModel(model, new TIntArrayList(uniqClasses));
        for (Node child : node.getChildren()) {
          final HierarchicalModel childModel = child.accept(this);
          if (childModel != null) {
            hierarchicalModel.addChild(childModel, child.id);
          }
        }
        return hierarchicalModel;
      }

      @Override
      public HierarchicalModel visit(final LeafNode node) {
        return null;
      }
    };
    return tree.getRoot().accept(learner);
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



//  final Stack<HierarchicalModel> stackModels = new Stack<>();
//  final Stack<InternalNode> stackNodes = new Stack<>();
//  stackNodes.push((InternalNode) tree.getRoot());
//
//  while (stackNodes.size() > 0) {
//    final InternalNode node = stackNodes.pop();
//    final TIntList uniqClasses = new TIntLinkedList();
//    for (Node child : node.getChildren()) {
//      uniqClasses.add(child.id);
//    }
//
//    for (int i = 0; i < learn.length(); i++) {
//      final int dsClassLabel = globalLoss.label(i);
//      final List<Node> children = node.getChildren();
//      weights[i] = 0;
//      localClasses[i] = -1;
//      for (int j = 0; j < children.size(); j++) {
//        final Node child = children.get(j);
//        if (tree.isFirstDescendantOfSecondOrEqual(dsClassLabel, child.id)) {
//          weights[i] = 1;
//          localClasses[i] = j;
//          break;
//        }
//      }
//    }
//
//    final BlockwiseWeightedLoss<BlockwiseMLLLogit> localWeightedLoss = new BlockwiseWeightedLoss<>(
//        new BlockwiseMLLLogit(new IntSeq(localClasses), learn), weights);
//    final MultiClassModel model = (MultiClassModel) weak.fit(learn, localWeightedLoss);
//    final HierarchicalModel hierarchicalModel = new HierarchicalModel(model.dirs(), new TIntArrayList(uniqClasses));
//    for (Node child : node.getChildren()) {
//      if (child instanceof InternalNode) {
//        stackNodes.push((InternalNode) child);
//      }
//    }
//  }
//
//
//  return null;
}





