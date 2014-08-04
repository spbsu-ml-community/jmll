package com.spbsu.ml.methods.hierarchical;

import com.spbsu.commons.seq.IntSeq;
import com.spbsu.commons.util.tree.FastTree;
import com.spbsu.commons.util.tree.Node;
import com.spbsu.commons.util.tree.NodeVisitor;
import com.spbsu.commons.util.tree.impl.node.InternalNode;
import com.spbsu.commons.util.tree.impl.node.LeafNode;
import com.spbsu.ml.*;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.tools.MCTools;
import com.spbsu.ml.loss.blockwise.BlockwiseMLLLogit;
import com.spbsu.ml.loss.blockwise.BlockwiseWeightedLoss;
import com.spbsu.ml.methods.VecOptimization;
//import com.spbsu.ml.models.HierJoinedBinClassModel;
import com.spbsu.ml.models.HierarchicalModel;
import com.spbsu.ml.models.JoinedBinClassModel;
import com.spbsu.ml.models.MultiClassModel;
import gnu.trove.list.TIntList;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.list.linked.TIntLinkedList;

import java.util.List;
import java.util.Stack;

/**
 * User: qdeee
 * Date: 10.04.14
 */
@Deprecated
public class HierarchicalRefinedClassification extends VecOptimization.Stub<BlockwiseMLLLogit> {
  protected final VecOptimization<TargetFunc> weak;
  protected final FastTree tree;

  public HierarchicalRefinedClassification(final VecOptimization<TargetFunc> weak, final FastTree tree) {
    this.weak = weak;
    this.tree = tree;
  }

  @Override
  public Trans fit(final VecDataSet learn, final BlockwiseMLLLogit globalLoss) {
    final HierarchicalModel hierJoinedBinClassModel = firstTraverse(learn, globalLoss);
    final HierarchicalModel hierarchicalModel = secondTraverse(learn, globalLoss, hierJoinedBinClassModel);
    return hierarchicalModel;
  }

  private HierarchicalModel firstTraverse(final VecDataSet learn, final BlockwiseMLLLogit globalLoss) {
    //avoiding a lot of allocations
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
          localClasses[i] = -1;
          for (int j = 0; j < children.size(); j++) {
            final Node child = children.get(j);
            if (tree.isFirstDescendantOfSecondOrEqual(dsClassLabel, child.id)) {
              localClasses[i] = j;
              break;
            }
          }
        }

        final Func[] models = new Func[uniqClasses.size()];
        for (int j = 0; j < uniqClasses.size(); j++) {
          final IntSeq oneVsRestTarget = MCTools.extractClassForBinary(new IntSeq(localClasses), j);
          final BlockwiseMLLLogit localLoss = new BlockwiseMLLLogit(oneVsRestTarget, learn);
          final MultiClassModel model = (MultiClassModel) weak.fit(learn, localLoss);
          models[j] = (Func) model.getInternModel().dirs()[0];
        }
        final HierarchicalModel nodeModel = new HierarchicalModel(new JoinedBinClassModel(models), uniqClasses);
        for (Node child : node.getChildren()) {
          final HierarchicalModel childModel = child.accept(this);
          if (childModel != null) {
            nodeModel.addChild(childModel, child.id);
          }
        }
        return nodeModel;
      }

      @Override
      public HierarchicalModel visit(final LeafNode node) {
        return null;
      }
    };
    return tree.getRoot().accept(learner);
  }

  private HierarchicalModel secondTraverse(final VecDataSet learn, final BlockwiseMLLLogit globalLoss, final HierarchicalModel cleanModel) {
    //avoiding a lot of allocations
    final int[] weights = new int[learn.length()];
    final int[] localClasses = new int[learn.length()];

    final Stack<HierarchicalModel> modelsStack = new Stack<>();
    modelsStack.push(cleanModel);

    final NodeVisitor<HierarchicalModel> learner = new NodeVisitor<HierarchicalModel>() {
      @Override
      public HierarchicalModel visit(final InternalNode node) {
        final TIntList uniqClasses = new TIntLinkedList();
        for (Node child : node.getChildren()) {
          uniqClasses.add(child.id);
        }

        //collect indices (what about previous model errors?)
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

        //learn and apply model to the sub(!)pool
        //
        final BlockwiseWeightedLoss<BlockwiseMLLLogit> localWeightedLoss = new BlockwiseWeightedLoss<>(
            new BlockwiseMLLLogit(new IntSeq(localClasses), learn),
            weights
        );
        final MultiClassModel dirtyModel = (MultiClassModel) weak.fit(learn, localWeightedLoss);
        //


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
////  private BFGrid grid;
//
//  public HierarchicalRefinedClassification(final int weakIters, final double weakStep) {
//    this.weakIters = weakIters;
//    this.weakStep = weakStep;
//  }
//
//  @Override
//  public Trans fit(final VecDataSet learn, final HierLoss hierLoss) {
////    grid = GridTools.medianGrid(learn, 32);
//    final HierJoinedBinClassModel hierJoinedBinClassModel = traverseFitBottomTop(hierLoss.getHierRoot());
//    final HierarchicalModel hierarchicalModel = traverseFitTopBottom(hierLoss.getHierRoot(), hierJoinedBinClassModel);
//    return hierarchicalModel;
//  }
//
//  private HierJoinedBinClassModel traverseFitBottomTop(final HierarchyTree.Node node) {
//    List<HierJoinedBinClassModel> childrenModels = new ArrayList<HierJoinedBinClassModel>(node.getChildren().size());
//    TIntList childrenModelsId = new TIntArrayList(node.getChildren().size());
//    for (HierarchyTree.Node child : node.getChildren()) {
//      if (!child.isLeaf()) {
//        childrenModels.add(traverseFitBottomTop(child));
//        childrenModelsId.add(child.getCategoryId());
//      }
//    }
//
//    final HierJoinedBinClassModel hierModel;
//    if (node.isTrainingNode()) {
//      final TIntList labels = new TIntArrayList();
//      for (HierarchyTree.Node child : node.getChildren()) {
//        labels.add(child.getCategoryId());
//      }
//      if (node.hasOwnDS())
//        labels.add(node.getCategoryId());
//
//      System.out.println(String.format("\n\nBoosting at node %d is started, DS size=%d", node.getCategoryId(), node.getEntriesCount()));
//      final Func[] resultModels = new Func[labels.size()];
//      for (int c = 0; c < labels.size(); c++) {
//        final VecDataSet childDS = node.createDSForChild(labels.get(c));
//        final LLLogit globalLoss = new LLLogit(node.createTargetForChild(c));
//
//        final int classIndex = labels.get(c);
//        final BFGrid grid = GridTools.medianGrid(childDS, 32);
//        final GradientBoosting<LLLogit> boosting = new GradientBoosting<LLLogit>(new GreedyObliviousTree<L2>(grid, 5), weakIters, weakStep);
//        final ProgressHandler calcer = new ProgressHandler() {
//          int iter = 0;
//
//          @Override
//          public void invoke(Trans partial) {
//            if ((iter + 1) % 20 == 0) {
//              double value = globalLoss.value(partial.transAll(DataTools.data(childDS)));
//              System.out.print(String.format("Node#%d, class=%d, positive examples=%d, iter=%d, LLLogit=%s\r",
//                  node.getCategoryId(), classIndex, MCTools.classEntriesCount(globalLoss, 1), iter, value));
//            }
//            iter++;
//          }
//        };
//        boosting.addListener(calcer);
//
//        Ensemble ensemble = boosting.fit(childDS, globalLoss);
//        System.out.println();
//        resultModels[c] = new FuncEnsemble(ArrayTools.map(ensemble.models, Func.class, new Computable<Trans, Func>() {
//          @Override
//          public Func compute(final Trans argument) {
//            return (Func)argument;
//          }
//        }), ensemble.weights);
//      }
//
//      hierModel = new HierJoinedBinClassModel(resultModels, labels);
//    }
//    else {
//      //this node has only one child, so we introduce max const func that will return this child with probability = 1
//      final Func[] resultModels = {
//          new Func.Stub() {
//            @Override
//            public double value(Vec x) {
//              return Double.MAX_VALUE;
//            }
//            @Override
//            public int dim() {
//              return 0;
//            }
//          }
//      };
//      final TIntArrayList classLabels = new TIntArrayList(new int[] {node.getChildren().get(0).getCategoryId()});
//      hierModel = new HierJoinedBinClassModel(resultModels, classLabels);
//    }
//    for (int i = 0; i < childrenModels.size(); i++) {
//      final HierJoinedBinClassModel child = childrenModels.get(i);
//      final int catId = childrenModelsId.get(i);
//      hierModel.addChildren(child, catId);
//    }
//    return hierModel;
//  }
//
//  private HierarchicalModel traverseFitTopBottom(final HierarchyTree.Node node, final HierJoinedBinClassModel btModel) {
//    final TIntList labels = new TIntArrayList(2);
//    final Func[] resultModels;
//
//    if (node.isTrainingNode()) {
//      final TIntList idxs = node.joinLists();
//      final VecDataSet ds = node.createDS();
//      final MLLLogit target = node.createTarget(labels);
//      final Vec predicted = btModel.bestClassAll(ds.data());
//      final TIntList errors = new TIntLinkedList();
//      for (int i = 0; i < predicted.dim(); i++) {
//        if (target.label(i) != predicted.get(i)) {
//          node.removeIdx(idxs.get(i));
//          errors.add(idxs.get(i));
//        }
//      }
//      if (errors.size() > 0) {
//        node.addErrorChild(errors);
//      }
//
//      final VecDataSet learnDS = node.createDS();
//      final MLLLogit globalLoss  = node.createTarget(labels);
//      final BFGrid grid = GridTools.medianGrid(learnDS, 32);
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
//            double value = globalLoss.value(partial.transAll(DataTools.data(learnDS)));
//            System.out.print("Node#" + node.getCategoryId() + ", iter=" + index + ", MLLLogitValue=" + value + "\r");
//          }
//          index++;
//        }
//      };
//      boosting.addListener(calcer);
//
//      System.out.println("\n\nBoosting at node " + node.getCategoryId() + " is started, learn DS size=" + learnDS.length() + ", filtered errors = " + errors.size());
//      {
//        for (HierarchyTree.Node childNode : node.getChildren()) {
//          System.out.println("entries for class{" + childNode.getCategoryId() + "} = " + ArrayTools.entriesCount(globalLoss.labels(), childNode.getCategoryId()));
//        }
//      }
//      final Ensemble ensemble = boosting.fit(learnDS, globalLoss);
//      System.out.println();
//      resultModels = MultiClassModel.joinBoostingResults(ensemble).dirs();
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
//    HierarchicalModel hierarchicalModel = new HierarchicalModel(resultModels, labels);
//
//    for (HierarchyTree.Node childNode : node.getChildren()) {
//      if (childNode.isLeaf())
//        continue;
//      final HierJoinedBinClassModel childBottomTopModel = btModel.getModelByLabel(childNode.getCategoryId());
//      assert childBottomTopModel != null;
//      final HierarchicalModel childTopBottomModel = traverseFitTopBottom(childNode, childBottomTopModel);
//      if (childTopBottomModel != null) {
//        hierarchicalModel.addChild(childTopBottomModel, childNode.getCategoryId());
//      }
//    }
//
//    return hierarchicalModel;
//  }

}
