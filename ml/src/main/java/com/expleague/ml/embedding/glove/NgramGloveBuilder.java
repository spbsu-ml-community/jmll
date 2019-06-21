package com.expleague.ml.embedding.glove;

import com.expleague.commons.math.MathTools;
import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.seq.CharSeq;
import com.expleague.commons.util.logging.Interval;
import com.expleague.ml.embedding.Embedding;
import com.expleague.ml.embedding.impl.EmbeddingImpl;
import com.expleague.ml.embedding.impl.ScoreCalculator;
import com.expleague.ml.embedding.impl.TrigramBasedBuilder;
import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.map.TObjectDoubleMap;
import gnu.trove.map.hash.TObjectDoubleHashMap;

import java.util.HashMap;
import java.util.Map;
import java.util.stream.IntStream;

public class NgramGloveBuilder extends TrigramBasedBuilder {
  private double xMax = 10;
  private double alpha = 0.75;
  private int dim = 50;
  private double TRIGRAM_STEP_DISCOUNT = 10d;

  public NgramGloveBuilder xMax(int xMax) {
    this.xMax = xMax;
    return this;
  }

  public NgramGloveBuilder alpha(double alpha) {
    this.alpha = alpha;
    return this;
  }

  public NgramGloveBuilder dim(int dim) {
    this.dim = dim;
    return this;
  }

  private double weightingFunc(double x) {
    return x < xMax ? Math.pow((x / xMax), alpha) : 1;
  }

  @Override
  public Embedding<CharSeq> fit() {
    final int vocab_size = dict().size();
    final int trigram_vocab_size = trigDict().size();

    final Mx leftVectors = new VecBasedMx(vocab_size, dim);
    final Mx rightVectors = new VecBasedMx(vocab_size, dim);
    final Mx trigramVectors = new VecBasedMx(trigram_vocab_size, dim);

    //final Vec biasLeft = new ArrayVec(vocab_size);
    //final Vec biasRight = new ArrayVec(vocab_size);

    final Mx softMaxLeft = new VecBasedMx(vocab_size, dim);
    final Mx softMaxRight = new VecBasedMx(vocab_size, dim);
    final Mx softMaxTrigram = new VecBasedMx(trigram_vocab_size, dim);
    VecTools.fill(softMaxLeft, 1.);
    VecTools.fill(softMaxRight, 1.);
    VecTools.fill(softMaxTrigram, 1.);

    //final Vec softBiasLeft = new ArrayVec(vocab_size);
    //final Vec softBiasRight = new ArrayVec(vocab_size);
    //VecTools.fill(softBiasLeft, 1.);
    //VecTools.fill(softBiasRight, 1.);

    for (int i = 0; i < vocab_size; i++) {
      //biasRight.set(i, initializeValue());
      //biasLeft.set(i, initializeValue());
      for (int j = 0; j < dim; j++) {
        leftVectors.set(i, j, initializeValue());
        rightVectors.set(i, j, initializeValue());
      }
    }
    for (int i = 0; i < trigram_vocab_size; i++) {
      for (int j = 0; j < dim; j++) {
        trigramVectors.set(i, j, initializeValue());
      }
    }

    TDoubleArrayList wordsProbabsLeft = new TDoubleArrayList(vocab_size);
    TDoubleArrayList wordsProbabsRight = new TDoubleArrayList(vocab_size);
    TDoubleArrayList trigramProbabs = new TDoubleArrayList(trigram_vocab_size);
    final double X_sum = countWordsProbabs(wordsProbabsLeft, wordsProbabsRight);
    final double Trig_sum = countTrigramProbabs(trigramProbabs);

    for (int iter = 0; iter < T(); iter++) {
      Interval.start();
      final ScoreCalculator scoreCalculator = new ScoreCalculator(vocab_size);
      IntStream.range(0, vocab_size).parallel().forEach(i -> {
        final CharSeq word_i = dict().get(i);
        final Vec leftWord = leftVectors.row(i);
        final Vec softMaxL = softMaxLeft.row(i);
        final Vec left = VecTools.copy(leftWord);

        /*final Vec left = new ArrayVec();
        final Vec[] leftComponents = new Vec[word.length() - 2];
        leftComponents[0] = leftVectors.row(i);
        for (int k = 0; k < word.length() - 3; k++) {
          leftComponents[k + 1] = trigram(word.sub(k, k + 3), trigramVectors);
        }
        Stream.of(leftComponents).forEach(v -> append(left, v));*/

        cooc(i, (j, X_ij) -> {
          final Vec right = rightVectors.row(j);
          final Vec softMaxR = softMaxRight.row(j);

          double trigramMutualInfo = 0d;
          for (int k = 0; k < word_i.length() - 2; k++) {
            final int ngram = trigrToId(word_i.subSequence(k, k + 3));
            trigramMutualInfo += Math.log(getTrigramCooc(ngram, j)) - Math.log(trigramProbabs.get(ngram)) - Math.log(wordsProbabsRight.get(j)) + Math.log(Trig_sum);
          }

          final double asum = VecTools.multiply(left, right);
          final double logMutualInfo = Math.log(X_ij) - Math.log(wordsProbabsLeft.get(i)) - Math.log(wordsProbabsRight.get(j)) + Math.log(X_sum);
          final double diff = asum - logMutualInfo - trigramMutualInfo;
          //final double diff = biasTr + biasLeft.get(i) + biasRight.get(j) * (word_i.length() - 1) + asum - Math.log(X_ij) * (word_i.length() - 1);
          final double weight = weightingFunc(X_ij);
          final double fdiff = step() * diff * weight;
          scoreCalculator.adjust(i, j, weight, 0.5 * weight * MathTools.sqr(diff));
          IntStream.range(0, dim).forEach(id -> {
            final double dL = fdiff * right.get(id);
            final double dTr = dL / TRIGRAM_STEP_DISCOUNT;
            final double dR = fdiff * left.get(id);
            leftWord.adjust(id, -dL / Math.sqrt(softMaxL.get(id)));
            right.adjust(id, -dR / Math.sqrt(softMaxR.get(id)));
            softMaxL.adjust(id, dL * dL);
            softMaxR.adjust(id, dR * dR);
            for (int k = 0; k < word_i.length() - 2; k++) {
              trigramVectors.row(k).adjust(id, -dTr / Math.sqrt(softMaxTrigram.get(id)));
              softMaxTrigram.row(k).adjust(id, dTr * dTr);
            }
          });

          //biasLeft.adjust(i, -fdiff / Math.sqrt(softBiasLeft.get(i)));
          //biasRight.adjust(j, -fdiff / Math.sqrt(softBiasRight.get(j)));
          //softBiasLeft.adjust(i, MathTools.sqr(fdiff));
          //softBiasRight.adjust(j, MathTools.sqr(fdiff));
        });
      });

      Interval.stopAndPrint("Iteration " + iter + ", score: " + scoreCalculator.gloveScore());
    }

    final Map<CharSeq, Vec> mapping = new HashMap<>();
    for (int i = 0; i < dict().size(); i++) {
      final CharSeq word = dict().get(i);
      mapping.put(word, VecTools.sum(leftVectors.row(i), rightVectors.row(i)));
    }

    return new EmbeddingImpl<>(mapping);
  }

  private double initializeValue() {
    return (Math.random() - 0.5) / dim;
  }
}
