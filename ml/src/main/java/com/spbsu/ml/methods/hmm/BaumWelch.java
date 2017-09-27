package com.spbsu.ml.methods.hmm;

import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.seq.Seq;
import com.spbsu.commons.seq.regexp.Alphabet;
import com.spbsu.commons.util.ThreadTools;
import com.spbsu.ml.data.set.DataSet;
import com.spbsu.ml.loss.LLLogit;
import com.spbsu.ml.methods.Optimization;
import com.spbsu.ml.models.hmm.HiddenMarkovModel;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.logging.Logger;

public class BaumWelch<T> implements Optimization<LLLogit,DataSet<Seq<T>>,Seq<T>> {
  private static final Logger log = Logger.getLogger(BaumWelch.class.getName());
  private final Alphabet<T> alphabet;
  private final int states;
  private int iterations;
  private FastRandom rng = new FastRandom(0);

  public BaumWelch(Alphabet<T> alphabet, int states, int iterations) {
    this.alphabet = alphabet;
    this.states = states;
    this.iterations = iterations;
  }

  @Override
  public HiddenMarkovModel<T> fit(DataSet<Seq<T>> learn, LLLogit llLogit) {
    final Vec[] betta = {
        new ArrayVec((states + 1) * states + states * alphabet.size()),
        new ArrayVec((states + 1) * states + states * alphabet.size())
    };

    VecTools.fill(betta[0], 1);
    VecTools.fillUniformPlus(betta[0].sub(states * (states + 1), states * alphabet.size()), rng, 1);

    final ThreadPoolExecutor bwCalcer = ThreadTools.createBGExecutor("BWCalcer", learn.length());
    final ThreadLocal<Mx> accBCache = ThreadLocal.withInitial(() -> new VecBasedMx(alphabet.size(), states));
    final ThreadLocal<Mx> ksiCache = ThreadLocal.withInitial(() -> new VecBasedMx(states, states));

    for (int t = 0; t < iterations; t++) {
      final Vec current = betta[t % 2];
      final Vec next = betta[(t + 1) % 2];

      normalizeBetta(current);

      final Mx A = new VecBasedMx(states, current.sub(states, states * states));
      final Mx B = new VecBasedMx(states,  current.sub(states * (states + 1), states * alphabet.size()));

      VecTools.fill(next, 0);
      final HiddenMarkovModel<T> hmm = new HiddenMarkovModel<>(alphabet, states, current);
      double[] ll = {0};
      double totalLength = 0;
      final CountDownLatch latch = new CountDownLatch(learn.length());
      for (int i = 0; i < learn.length(); i++) {
        final Seq<T> seq = learn.at(i);
        if (seq.length() == 0 || llLogit.label(i) > 0) {
          latch.countDown();
          continue;
        }
        totalLength += seq.length();

        bwCalcer.execute(() -> {
          final Mx ksi = ksiCache.get();
          VecTools.fill(ksi, 0);

          final Mx forward = hmm.forward(seq);
          final Mx backward = hmm.backward(seq);

          for (int k = 0; k < seq.length() - 1; k++) {
            final int nextIdx = alphabet.index(seq, k + 1);
            double sum = 0;
            for (int u = 0; u < states; u++) {
              for (int v = 0; v < states; v++) {
                sum += forward.get(k, u) * A.get(u, v) * backward.get(k + 1, v) * B.get(nextIdx, v);
              }
            }
            if (sum < MathTools.EPSILON)
              System.out.println();
            for (int u = 0; u < states; u++) {
              for (int v = 0; v < states; v++) {
                final double increment = forward.get(k, u) * A.get(u, v) * backward.get(k + 1, v) * B.get(nextIdx, v);
                ksi.adjust(u, v, increment / sum);
              }
            }
          }

          //noinspection UnnecessaryLocalVariable
          final Mx distrib = forward;
          VecTools.scale(distrib, backward);
          final Vec sum = new ArrayVec(states);
          double llLocal = 0;
          for (int k = 0; k < seq.length(); k++) {
            final Vec states = distrib.row(k);
            VecTools.normalizeL1(states);
            llLocal += Math.log(VecTools.multiply(states, B.row(alphabet.index(seq, k))));
            VecTools.append(sum, states);
          }

          for (int u = 0; u < states; u++) {
            VecTools.normalizeL1(ksi.row(u));
            sum.set(u, 1. / (sum.get(u) + 1e-6));
          }

          final Mx accB = accBCache.get();
          VecTools.fill(accB, 0);

          for (int k = 0; k < seq.length(); k++) {
            final int nextIdx = alphabet.index(seq, k);
            final Vec bRow = accB.row(nextIdx);
            final Vec gamma = distrib.row(k);
            VecTools.scale(gamma, sum);
            VecTools.append(bRow, gamma);
          }

          synchronized (this) {
            ll[0] += llLocal;
            VecTools.incscale(next.sub(0, states), distrib.row(0), 1. / learn.length());
            VecTools.incscale(next.sub(states, states * states), ksi, 1. / learn.length());
            VecTools.incscale(next.sub((states + 1) * states, states * alphabet.size()), accB, 1. / learn.length());
          }
          latch.countDown();
        });
      }
      try {
        latch.await();
      }
      catch (InterruptedException e) {
        throw new RuntimeException(e);
      }
      log.fine("It: " + t + " unit perplexity: " + Math.exp(ll[0]/totalLength));
      System.out.println("It: " + t + " unit perplexity: " + Math.exp(ll[0]/totalLength));
    }

    return new HiddenMarkovModel<>(alphabet, states, betta[iterations % 2]);
  }

  private void normalizeBetta(Vec betta) {
    for (int i = 0; i < (states + 1) * states; i += states) {
      final Vec vec = betta.sub(i, states);
      VecTools.normalizeL1(vec);
    }

    final Mx B = new VecBasedMx(states,  betta.sub(states * (states + 1), states * alphabet.size()));
    final ArrayVec unit = new ArrayVec(alphabet.size());
    VecTools.fill(unit, 1);
    for (int j = 0; j < states; j++) {
      final Vec vec = B.col(j);
      VecTools.incscale(vec, unit, 1e-4);
      VecTools.normalizeL1(vec);
    }
  }
}
