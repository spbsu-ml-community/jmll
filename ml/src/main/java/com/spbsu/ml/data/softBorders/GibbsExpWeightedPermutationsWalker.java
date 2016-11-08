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
  private final double[] localRankWeights;
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
    this.rankWeights = rankWeights;
    this.lambda = lambda;

    this.blockPermutations = new ArrayList<>();
    Combinatorics.Permutations permutations = new Combinatorics.Permutations(blockSize);
    while (permutations.hasNext()) {
      blockPermutations.add(permutations.next());
    }

    this.localWeights = new double[blockPermutations.size()];
    this.localRanks = new int[blockSize];
    if (this.rankWeights != null) {
      this.localRankWeights = new double[blockSize];
    } else {
      this.localRankWeights = null;
    }
  }

  private void sampleBlock(int start) {

    for (int i = 0; i < blockSize; ++i) {
      localRanks[i] = cursor[start + i];
      if (rankWeights != null) {
        localRankWeights[i] = rankWeights[start + i];
      }
    }


    double totalWeight = 0;

    for (int idx = 0; idx < blockPermutations.size(); ++idx) {
      final int[] permutation = blockPermutations.get(idx);

      double permutationWeight = 0;
      for (int j = 0; j < blockSize; ++j) {
        final double rk = localRanks[permutation[j]];
        final double weight = localRankWeights != null ? localRankWeights[permutation[j]] : 1.0;
        permutationWeight += weight * Math.abs(rk - start - j);
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
      if (rankWeights != null) {
        assert(localRankWeights != null);
        rankWeights[start + i] = localRankWeights[permutation[i]];
      }
    }
  }

  public int[] sample() {
    for (int iter = 0; iter < cursor.length; ++iter) {
      sampleBlock(random.nextInt(cursor.length - blockSize + 1));
    }
    return cursor;
  }



  public static void main(final String[] args) {
    GibbsExpWeightedPermutationsWalker permutations = new GibbsExpWeightedPermutationsWalker(200, 2);
    for (int i = 0; i < 50; ++i) {
      System.out.println(Arrays.toString(permutations.sample()));
    }
  }

}
