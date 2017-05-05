package com.spbsu.ml.data.softBorders;

import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.commons.util.Combinatorics;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by noxoomo on 06/11/2016.
 */

public class GibbsExpWeightedPermutationsWalker implements Sampler<int[]> {
  private final int blockSize;
  private final double lambda;
  private final List<int[]> blockPermutations;
  private final double[] localWeights;
  private final int[] localRanks;
  private final FastRandom random = new FastRandom();

  private final double[] rankWeights;
  private final int[] cursor;

  public GibbsExpWeightedPermutationsWalker(final int n,
                                            final double lambda) {
    this(n, lambda, null);
  }

  public GibbsExpWeightedPermutationsWalker(final int n,
                                            final double lambda,
                                            final double[] rankWeights) {

    this.blockSize = Math.min(4, n);

    this.cursor = ArrayTools.sequence(0, n);
    if (rankWeights != null) {
      assert (rankWeights.length == n);
      this.rankWeights = rankWeights;
    } else {
      this.rankWeights = new double[n];
      ArrayTools.fill(this.rankWeights, 1.0);
    }
    for (int i = 1; i < n; ++i) {
      this.rankWeights[i] += this.rankWeights[i - 1];
    }
    this.lambda = lambda;

    this.blockPermutations = new ArrayList<>();
    Combinatorics.Permutations permutations = new Combinatorics.Permutations(blockSize);
    while (permutations.hasNext()) {
      blockPermutations.add(permutations.next());
    }

    this.localWeights = new double[blockPermutations.size()];
    this.localRanks = new int[blockSize];
  }

  private void sampleBlock(int start) {
    System.arraycopy(cursor, start, localRanks, 0, blockSize);

    double totalWeight = 0;
    for (int idx = 0; idx < blockPermutations.size(); ++idx) {
      final int[] permutation = blockPermutations.get(idx);

      double permutationWeight = 0;

      for (int j = 0; j < blockSize; ++j) {
        final int rk = localRanks[permutation[j]];
        final double weightDiff = Math.abs(rankWeights[rk] - rankWeights[start + j]);
        permutationWeight += weightDiff;
      }
      permutationWeight *= lambda;
      localWeights[idx] = Math.exp(-permutationWeight);
      totalWeight += localWeights[idx];
    }

    double gain = random.nextDouble() * totalWeight;
    int takenIdx = -1;
    while (gain > 0) {
      gain -= localWeights[++takenIdx];
    }

    final int[] permutation = blockPermutations.get(takenIdx);
    for (int i = 0; i < blockSize; ++i) {
      cursor[start + i] = localRanks[permutation[i]];
    }
  }

  public int[] sample() {
    for (int iter = 0; iter < cursor.length; ++iter) {
      sampleBlock(random.nextInt(cursor.length - blockSize + 1));
    }
    return cursor;
  }

  public static void main(final String[] args) {
    final GibbsExpWeightedPermutationsWalker permutations = new GibbsExpWeightedPermutationsWalker(20, 0.5);
    for (int i = 0; i < 50; ++i) {
      System.out.println(Arrays.toString(permutations.sample()));
    }
  }
}
