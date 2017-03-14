package com.spbsu.ml.methods.multiclass.hierarchical;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.seq.IntSeq;
import com.spbsu.commons.util.Pair;
import com.spbsu.commons.util.tree.IntTree;
import com.spbsu.commons.util.tree.IntTreeVisitor;
import com.spbsu.commons.math.Func;
import com.spbsu.commons.math.Trans;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.tools.DataTools;
import com.spbsu.ml.data.tools.MCTools;
import com.spbsu.ml.loss.LLLogit;
import com.spbsu.ml.loss.blockwise.BlockwiseMLLLogit;
import com.spbsu.ml.methods.VecOptimization;
import com.spbsu.ml.models.multiclass.HierarchicalModel;
import com.spbsu.ml.models.multiclass.JoinedBinClassModel;
import com.spbsu.ml.models.multiclass.MCModel;
import gnu.trove.iterator.TIntIterator;
import gnu.trove.list.TIntList;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.list.linked.TIntLinkedList;

import java.util.Arrays;
import java.util.Stack;

/**
 * User: qdeee
 * Date: 10.04.14
 */

public class HierarchicalRefinedClassification extends VecOptimization.Stub<BlockwiseMLLLogit> {
  protected final VecOptimization<LLLogit> weakBinClass;
  protected final VecOptimization<BlockwiseMLLLogit> weakMultiClass;
  protected final IntTree tree;

  public HierarchicalRefinedClassification(final VecOptimization<LLLogit> weakBinClass,
                                           final VecOptimization<BlockwiseMLLLogit> weakMultiClass,
                                           final IntTree tree) {
    this.weakBinClass = weakBinClass;
    this.weakMultiClass = weakMultiClass;
    this.tree = tree;
  }

  @Override
  public Trans fit(final VecDataSet learn, final BlockwiseMLLLogit globalLoss) {
    final SpecialHierModel hierJoinedBinClassModel = firstTraverse(learn, globalLoss);
    final HierarchicalModel hierarchicalModel = secondTraverse(learn, globalLoss, hierJoinedBinClassModel);
    return hierarchicalModel;
  }

  private SpecialHierModel firstTraverse(final VecDataSet learn, final BlockwiseMLLLogit globalLoss) {
    final int[] localClasses = new int[learn.length()];
    final Stack<SpecialHierModel> modelsStack = new Stack<>();

    final IntTreeVisitor visitor = new IntTreeVisitor() {
      @Override
      public void visit(final int node) {
        final TIntList uniqClasses = new TIntLinkedList();
        {
          final TIntIterator children = tree.getChildren(node);
          while (children.hasNext()) {
            uniqClasses.add(children.next());
          }
        }

        for (int i = 0; i < learn.length(); i++) {
          final int dsClassLabel = globalLoss.label(i);
          localClasses[i] = -1;
          final TIntIterator children = tree.getChildren(node);
          for (int j = 0; children.hasNext(); j++) {
            final int child = children.next();
            if (dsClassLabel == child || tree.isDescendant(dsClassLabel, child)) {
              localClasses[i] = j;
              break;
            }
          }
        }

        final Func[] models = new Func[uniqClasses.size()];
        for (int j = 0; j < uniqClasses.size(); j++) {
          final VecDataSet dsForLearn;
          final Vec targetForLearn;

          final Vec oneVsRestTarget = MCTools.extractClassForBinary(new IntSeq(localClasses), j);
          if (node != tree.ROOT) {
            final TIntList dsIdxs = new TIntLinkedList();
            for (int i = 0; i < learn.length(); i++) {
              if (localClasses[i] == -1 || localClasses[i] == j)
                dsIdxs.add(i); //everyone exclude siblings
            }

            final Pair<VecDataSet, Vec> pair = DataTools.createSubset(learn, oneVsRestTarget, dsIdxs.toArray());
            dsForLearn = pair.first;
            targetForLearn = pair.second;
          }
          else {
            dsForLearn = learn;
            targetForLearn = oneVsRestTarget;
          }
          models[j] = (Func) weakBinClass.fit(dsForLearn, new LLLogit(targetForLearn, learn));
        }

        final SpecialHierModel nodeModel = new SpecialHierModel(new JoinedBinClassModel(models), uniqClasses);
        modelsStack.push(nodeModel);

        final TIntIterator children = tree.getChildren(node);
        while (children.hasNext()) {
          final int child = children.next();
          if (tree.hasChildren(child)) {
            tree.accept(this, child);
            nodeModel.addChild(modelsStack.pop(), child);
          }
        }
      }
    };
    tree.accept(visitor, tree.ROOT);
    return modelsStack.pop();
  }

  private HierarchicalModel secondTraverse(final VecDataSet learn, final BlockwiseMLLLogit globalLoss, final SpecialHierModel firstModel) {
    final int[] localClasses = new int[learn.length()];

    final Stack<int[]> errorsStack = new Stack<>();

    final Stack<SpecialHierModel> cleanModelsStack = new Stack<>();
    final Stack<HierarchicalModel> refinedModelsStack = new Stack<>();
    cleanModelsStack.push(firstModel);

    final IntTreeVisitor visitor = new IntTreeVisitor() {
      @Override
      public void visit(final int node) {
        final int[] errors = errorsStack.size() > 0 ? Arrays.copyOf(errorsStack.peek(), learn.length()) : new int[learn.length()];

        final TIntList uniqClasses = new TIntLinkedList();
        {
          final TIntIterator children = tree.getChildren(node);
          while (children.hasNext()) {
            uniqClasses.add(children.next());
          }
        }

        final SpecialHierModel cleanModel = cleanModelsStack.pop();

        final TIntList dsIdxs = new TIntLinkedList();
        for (int i = 0; i < learn.length(); i++) {
          localClasses[i] = -1;
          if (errors[i] == 1) {
            continue; //skip errors from top
          }

          final int dsClassLabel = globalLoss.label(i);
          final TIntIterator children = tree.getChildren(node);
          for (int j = 0; children.hasNext(); j++) {
            final int child = children.next();
            if (dsClassLabel == child || tree.isDescendant(dsClassLabel, child)) {
              dsIdxs.add(i);
              localClasses[i] = j;
              break;
            }
          }

          if (dsIdxs.get(dsIdxs.size() - 1) == i) {
            final int predictedLocalClass = cleanModel.bestClass(learn.at(i));
            if (localClasses[i] != predictedLocalClass) {
              localClasses[i] = uniqClasses.size();
              errors[i] = 1; //for next levels
            }
          }
        }

        if (MCTools.classEntriesCount(new IntSeq(localClasses), uniqClasses.size()) > 0) {
          uniqClasses.add(-1);
        }

        final Pair<VecDataSet, IntSeq> pair = DataTools.createSubset(learn, new IntSeq(localClasses), dsIdxs.toArray());
        final MCModel model = (MCModel) weakMultiClass.fit(pair.first, new BlockwiseMLLLogit(pair.second, learn));
        final HierarchicalModel refinedModel = new HierarchicalModel(model, new TIntArrayList(uniqClasses));

        refinedModelsStack.push(refinedModel); //for top levels
        errorsStack.push(errors);              //for bottom levels

        final TIntIterator children = tree.getChildren(node);
        while (children.hasNext()) {
          final int child = children.next();
          if (tree.hasChildren(child)) {
            cleanModelsStack.push((SpecialHierModel) cleanModel.getChild(child));
            tree.accept(this, child);
            refinedModel.addChild(refinedModelsStack.pop(), child);
          }
        }

        errorsStack.pop();
      }
    };
    tree.accept(visitor, tree.ROOT);
    return refinedModelsStack.pop();
  }

  private static class SpecialHierModel extends HierarchicalModel {
    public SpecialHierModel(final JoinedBinClassModel basedOn, final TIntList classLabels) {
      super(basedOn, classLabels);
    }

    //we need to accumulate signals from bottom levels
    private Vec deepTrans(final Vec x) {
      final Vec trans = ((JoinedBinClassModel) basedOn).getInternModel().trans(x);
      for (int i = 0; i < classLabels.size(); i++) {
        final int label = classLabels.get(i);
        final SpecialHierModel model = (SpecialHierModel) label2childModel.get(label);
        if (model != null) {
          final double val = VecTools.sum(model.deepTrans(x));
          trans.adjust(i, val);
        }
      }
      return trans;
    }

    @Override
    public int bestClass(final Vec x) {
      final Vec deepTrans = deepTrans(x);
      return VecTools.argmax(deepTrans); //return local best class index (not label)
    }
  }
}