package com.spbsu.ml.data.tools;

import com.spbsu.commons.seq.IntSeq;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.commons.util.tree.IntArrayTree;
import com.spbsu.commons.util.tree.IntTree;
import gnu.trove.list.TIntList;
import gnu.trove.list.linked.TIntLinkedList;
import gnu.trove.map.TIntIntMap;
import gnu.trove.map.hash.TIntIntHashMap;

/**
* User: qdeee
* Date: 07.04.14
*/
public class HierTools {
  public static class TreeBuilder {
    private final int minEntries;
    private final TIntList pruned = new TIntLinkedList();

    private IntTree tree = new IntArrayTree();
    private TIntIntMap targetMapping = new TIntIntHashMap();

    public TreeBuilder(final int minEntries) {
      this.minEntries = minEntries;
    }

    public void createFromOrderedMulticlass(final IntSeq targetMC) {
      tree = new IntArrayTree();
      targetMapping = new TIntIntHashMap();
      pruned.clear();

      final int countClasses = MCTools.countClasses(targetMC);
      final int[] counts = new int[countClasses];
      for (int i = 0; i < targetMC.length(); i++) {
        counts[targetMC.at(i)]++;
      }

      splitAndCreate(counts, 0, counts.length, 0);

      final int[] leaves = tree.leaves(IntTree.TRAVERSE_STRATEGY.DEPTH_FIRST);
      for (int i = 0; i < pruned.size(); i++) {
        final int start = i > 0 ? pruned.get(i - 1) : 0;
        final int end = pruned.get(i);
        for (int j = start; j < end; j++) {
          targetMapping.put(j, leaves[i]);
        }
      }
    }

    private void splitAndCreate(final int[] counts, final int from, final int to, final int parentNode) {
      if (to - from == 1) {
        return;
      }

      final int sum = ArrayTools.sum(counts, from, to);

      int bestSplit = -1;
      int minSubtract = Integer.MAX_VALUE;
      int curSum = 0;
      for (int split = from; split < to - 1; split++) {
        curSum += counts[split];
        final int subtract = Math.abs((sum - curSum) - curSum);
        if (subtract < minSubtract) {
          minSubtract = subtract;
          bestSplit = split;
        }
      }

      final int sum1 = ArrayTools.sum(counts, from, bestSplit + 1);
      final int sum2 = ArrayTools.sum(counts, bestSplit + 1, to);

      if (sum1 > minEntries && sum2 > minEntries) {
        final int leftChild = tree.addTo(parentNode);
        if (bestSplit > from)
          splitAndCreate(counts, from, bestSplit + 1, leftChild);
        else
          pruned.add(bestSplit + 1);

        final int rightChild = tree.addTo(parentNode);
        if (bestSplit < to - 2)
          splitAndCreate(counts, bestSplit + 1, to, rightChild);
        else
          pruned.add(to);
      }
      else {
        pruned.add(to);
      }
    }

    public IntTree releaseTree() {
      return tree;
    }

    public TIntIntMap releaseMapping() {
      return targetMapping;
    }
  }
}
