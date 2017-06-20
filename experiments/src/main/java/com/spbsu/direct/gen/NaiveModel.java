package com.spbsu.direct.gen;

import com.spbsu.commons.io.codec.seq.Dictionary;
import com.spbsu.commons.math.vectors.impl.vectors.SparseVec;
import com.spbsu.commons.seq.CharSeq;
import com.spbsu.commons.seq.IntSeq;
import gnu.trove.list.TIntList;

import static java.lang.Math.log;

public class NaiveModel {
  private final Dictionary<CharSeq> dict;
  private final NaiveProvider[] providers;

  public double totalFreq = 0;
  public final TIntList freqs;


  public NaiveModel(final Dictionary<CharSeq> dict,
                    final TIntList freqsLA) {
    this.dict = dict;
    this.providers = new NaiveProvider[dict.size() + 1];

    for (int i = 0; i < this.providers.length; ++i) {
      this.providers[i] = new NaiveProvider(dict);
    }

    this.freqs = freqsLA;
    this.totalFreq = freqsLA.sum();
  }

  public void processSeq(final IntSeq prevQSeq) {
    for (int i = 0; i < prevQSeq.length(); ++i) {
      int symbol = prevQSeq.intAt(i);

      // TODO: check useless and remove
      if (freqs.size() < symbol) {
        freqs.fill(freqs.size(), symbol + 1, 0);
      }

      freqs.set(symbol, freqs.get(symbol) + 1);
      totalFreq++;
    }
  }

  public void processGeneration(final IntSeq prevQSeq,
                                final IntSeq currentQSeq) {
    final int mask = (1 << currentQSeq.length()) - 1;

    for (int i = 0; i < prevQSeq.length(); ++i) {
      // TODO: check useless and remove
      final int index = prevQSeq.intAt(i);
      if (index < 0) {
        continue;
      }

      providers[index].apply(mask, currentQSeq);
    }

    providers[dict.size()].apply(mask, currentQSeq);
  }

  public double maxLogP(final IntSeq prevQSeq,
                        final IntSeq currentQSeq) {
    if (prevQSeq.length() * currentQSeq.length() > 10) {
      // too many variants of bipartite graph
      return Double.NEGATIVE_INFINITY;
    }

    final int variantsCount = 1 << (prevQSeq.length() * currentQSeq.length());
    final int mask = (1 << currentQSeq.length()) - 1;

    double maxLogProbability = Double.NEGATIVE_INFINITY;

    for (int currVariant = 0; currVariant < variantsCount; ++currVariant) {
      double logProbability = 0;

      int variant = currVariant;
      int generated = 0;

      for (int i = 0; i < prevQSeq.length(); ++i, variant >>= currentQSeq.length()) {
        final int fragment = variant & mask;
        generated |= fragment;

        // TODO: check useless and remove
        final int index = prevQSeq.intAt(i);
        if (index < 0) {
          continue;
        }

        logProbability += providers[index].logP(fragment, currentQSeq);
      }

      logProbability += providers[dict.size()].logP((~generated) & mask, currentQSeq);

      maxLogProbability = Math.max(maxLogProbability, logProbability);
    }

    return maxLogProbability;
  }

  private static class NaiveProvider {
    private final Dictionary<CharSeq> dict;
    private final SparseVec freqs;
    private Integer totalFreq;

    public NaiveProvider(final Dictionary<CharSeq> dict) {
      this.dict = dict;
      this.freqs = new SparseVec(dict.size() + 1, 0);
      this.totalFreq = 0;
    }

    public void apply(int mask, final IntSeq seq) {
      for (int i = 0; i < seq.length(); ++i, mask >>= 1) {
        if ((mask & 1) == 1) {
          freqs.adjust(seq.intAt(i), 1);
          ++totalFreq;
        }
      }

      // EMPTY word
      freqs.adjust(dict.size(), 1);
      ++totalFreq;
    }

    public double logP(int mask, final IntSeq seq) {
      if (mask == 0) {
        return log(1.0 * (freqs.get(dict.size()) + 1) / (totalFreq + dict.size() + 1));
      }

      double probability = 0;

      for (int i = 0; i < seq.length(); ++i, mask >>= 1) {
        if ((mask & 1) == 1) {
          probability += log(1.0 * (freqs.get(seq.intAt(i)) + 1) / (totalFreq + dict.size() + 1));
        }
      }

      return probability;
    }
  }
}