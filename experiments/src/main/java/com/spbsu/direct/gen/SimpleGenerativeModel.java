package com.spbsu.direct.gen;

import com.spbsu.commons.func.Action;
import com.spbsu.commons.io.StreamTools;
import com.spbsu.commons.io.codec.seq.Dictionary;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.seq.*;
import gnu.trove.list.TIntList;

import java.io.IOException;
import java.io.Writer;

import static java.lang.Math.exp;

/**
 * User: solar
 * Date: 12.11.15
 * Time: 11:33
 */
public class SimpleGenerativeModel {
  public static final String EMPTY_ID = "##EMPTY##";
  public static final int GIBBS_COUNT = 5;

  private final WordGenProbabilityProvider[] providers;
  private final Dictionary<CharSeq> dict;
  private final FastRandom rng = new FastRandom(0);

  public double totalFreq = 0;
  public final TIntList freqs;

  public SimpleGenerativeModel(final Dictionary<CharSeq> dict,
                               final TIntList freqsLA) {
    this.dict = dict;
    this.providers = new WordGenProbabilityProvider[dict.size() + 1]; // +1 -- for EMPTY word

    for (int i = 0; i < providers.length; ++i) {
      this.providers[i] = new WordGenProbabilityProvider(i, dict);
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
                                final IntSeq currentQSeq,
                                final double alpha) {
    if (prevQSeq.length() * currentQSeq.length() > 10) {
      // too many variants of bipartite graph
      return;
    }

    final int variantsCount = 1 << (prevQSeq.length() * currentQSeq.length());
    final int mask = (1 << currentQSeq.length()) - 1;

    final Vec weights = new ArrayVec(variantsCount);

    for (int currVariant = 0; currVariant < variantsCount; ++currVariant) {
      double variantLogProBab = 0;

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

        variantLogProBab += providers[index].logP(fragment, currentQSeq);
      }

      variantLogProBab += providers[dict.size()].logP((~generated) & mask, currentQSeq);

      // Gibbs
      weights.set(currVariant, Math.max(variantLogProBab, -50));
    }

    { // Gibbs
      double sum = 0;
      double normalizer = weights.get(0);

      for (int i = 0; i < variantsCount; ++i) {
        weights.set(i, exp(weights.get(i) - normalizer));
        sum += weights.get(i);
      }

      for (int i = 0; i < GIBBS_COUNT; ++i) {
        final int bestVariant = rng.nextSimple(weights, sum);
        applyGeneration(prevQSeq, currentQSeq, alpha / GIBBS_COUNT, bestVariant);
      }
    }
  }

  private void applyGeneration(final IntSeq prevQSeq,
                               final IntSeq currentQSeq,
                               final double alpha,
                               int bestVariant) {
    final int mask = (1 << currentQSeq.length()) - 1;

    int generated = 0;

    for (int i = 0; i < prevQSeq.length(); ++i, bestVariant >>= currentQSeq.length()) {
      final int fragment = bestVariant & mask;
      generated |= fragment;

      // TODO: check useless and remove
      final int index = prevQSeq.intAt(i);
      if (index < 0) {
        continue;
      }

      providers[index].update(fragment, currentQSeq, alpha);
    }

    providers[dict.size()].update((~generated) & mask, currentQSeq, alpha);
  }

  public void printProviders(final Writer out,
                             final boolean limit) {
    for (int i = 0; i < providers.length; ++i) {
      providers[i].print(out, limit);
    }
  }

  public void load(String inputFile) throws IOException {
    CharSeqTools.processLines(StreamTools.openTextFile(inputFile), new Action<CharSequence>() {
      final StringBuilder builder = new StringBuilder();

      public void invoke(CharSequence line) {
        if (line.equals("}")) {
          WordGenProbabilityProvider provider = new WordGenProbabilityProvider(builder.toString(), dict);
          providers[provider.providerIndex] = provider;
          builder.delete(0, builder.length());
        } else {
          builder.append(line);
        }
      }
    });
  }
}
