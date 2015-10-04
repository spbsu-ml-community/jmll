package com.spbsu.ml.cli.cv;

import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.commons.util.Pair;
import com.spbsu.ml.data.tools.DataTools;
import com.spbsu.ml.data.tools.Pool;
import com.spbsu.ml.data.tools.SubPool;

/**
 * User: qdeee
 * Date: 16.09.15
 */
public class FoldsEnumerator {
  private final Pool<?> sourcePool;
  private final int foldsCount;

  private int[][] foldIndices;
  private int currentFold = 0;

  public FoldsEnumerator(final Pool<?> sourcePool, final FastRandom random, final int foldsCount) {
    this.sourcePool = sourcePool;
    this.foldsCount = foldsCount;

    final double[] probs = ArrayTools.fill(new double[foldsCount], 1. / foldsCount);
    foldIndices = DataTools.splitAtRandom(sourcePool.size(), random, probs);
  }

  public int getFoldsCount() {
    return foldsCount;
  }

  public boolean hasNext() {
    return currentFold < foldsCount;
  }

  public Pair<? extends Pool, ? extends Pool> next() {
    final int[] learnIndices = getLearnIndices();
    final int[] testIndices = foldIndices[currentFold];
    currentFold++;
    return Pair.create(new SubPool(sourcePool, learnIndices), new SubPool(sourcePool, testIndices));
  }

  private int[] getLearnIndices() {
    final int learnSize = sourcePool.size() - foldIndices[currentFold].length;
    final int[] learnIndices = new int[learnSize];
    int currentTotalLength = 0;
    for (int i = 0; i < currentFold; i++) {
      final int foldLength = foldIndices[i].length;
      System.arraycopy(foldIndices[i], 0, learnIndices, currentTotalLength, foldLength);
      currentTotalLength += foldLength;
    }
    for (int i = currentFold + 1; i < foldIndices.length; i++) {
      final int foldLength = foldIndices[i].length;
      System.arraycopy(foldIndices[i], 0, learnIndices, currentTotalLength, foldLength);
      currentTotalLength += foldLength;
    }
    return learnIndices;
  }
}
