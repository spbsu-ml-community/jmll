package com.spbsu.ml.methods.multiclass.hierarchical;

import com.spbsu.commons.seq.IntSeq;
import com.spbsu.commons.util.Pair;
import com.spbsu.commons.util.tree.IntTree;
import com.spbsu.commons.util.tree.IntTreeVisitor;
import com.spbsu.commons.math.Trans;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.tools.DataTools;
import com.spbsu.ml.loss.blockwise.BlockwiseMLLLogit;
import com.spbsu.ml.methods.VecOptimization;
import com.spbsu.ml.models.MultiClassModel;
import com.spbsu.ml.models.multiclass.HierarchicalModel;
import gnu.trove.iterator.TIntIterator;
import gnu.trove.list.TIntList;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.list.linked.TIntLinkedList;

import java.util.Stack;

/**
 * User: qdeee
 * Date: 06.02.14
 */
public class HierarchicalClassification extends VecOptimization.Stub<BlockwiseMLLLogit> {
  protected final VecOptimization<BlockwiseMLLLogit> weakMultiClass;
  protected final IntTree tree;

  public HierarchicalClassification(final VecOptimization<BlockwiseMLLLogit> weakMultiClass, final IntTree tree) {
    this.weakMultiClass = weakMultiClass;
    this.tree = tree;
  }

  @Override
  public Trans fit(final VecDataSet learn, final BlockwiseMLLLogit globalLoss) {
    final int[] localClasses = new int[learn.length()];
    final Stack<HierarchicalModel> modelsStack = new Stack<HierarchicalModel>();

    final IntTreeVisitor visitor = new IntTreeVisitor() {
      @Override
      public void visit(final int node) {
        final TIntList dsIdxs = new TIntLinkedList();

        for (int i = 0; i < learn.length(); i++) {
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
        }

        final Pair<VecDataSet, IntSeq> pair = DataTools.createSubset(learn, new IntSeq(localClasses), dsIdxs.toArray());
        final MultiClassModel model = (MultiClassModel) weakMultiClass.fit(pair.first, new BlockwiseMLLLogit(pair.second, learn));

        final TIntList labels = new TIntLinkedList();
        final TIntIterator childrenIter = tree.getChildren(node);
        while (childrenIter.hasNext()) {
          labels.add(childrenIter.next());
        }

        final HierarchicalModel hierarchicalModel = new HierarchicalModel(model, new TIntArrayList(labels));

        modelsStack.push(hierarchicalModel);
        final TIntIterator children = tree.getChildren(node);
        while (children.hasNext()) {
          final int child = children.next();
          if (tree.hasChildren(child)) {
            tree.accept(this, child);
            hierarchicalModel.addChild(modelsStack.pop(), child);
          }
        }
      }
    };
    tree.accept(visitor, tree.ROOT);
    return modelsStack.pop();
  }
}