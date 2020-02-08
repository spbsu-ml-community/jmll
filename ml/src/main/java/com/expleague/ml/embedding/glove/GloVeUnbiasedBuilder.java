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
import com.expleague.ml.embedding.impl.CoocBasedBuilder;
import com.expleague.ml.embedding.impl.EmbeddingImpl;
import com.expleague.ml.embedding.impl.ScoreCalculator;
import gnu.trove.list.array.TDoubleArrayList;

import java.util.HashMap;
import java.util.Map;
import java.util.stream.IntStream;

public class GloVeUnbiasedBuilder extends CoocBasedBuilder {
  private double xMax = 10;
  private double alpha = 0.75;
  private int dim = 50;

  public GloVeUnbiasedBuilder xMax(int xMax) {
    this.xMax = xMax;
    return this;
  }

  public GloVeUnbiasedBuilder alpha(double alpha) {
    this.alpha = alpha;
    return this;
  }

  public GloVeUnbiasedBuilder dim(int dim) {
    this.dim = dim;
    return this;
  }

  private double weightingFunc(double x) {
    return x < xMax ? Math.pow((x / xMax), alpha) : 1;
  }

  @Override
  public Embedding<CharSeq> fit() {
    final int vocab_size = dict().size();
    final Mx leftVectors = new VecBasedMx(vocab_size, dim);
    final Mx rightVectors = new VecBasedMx(vocab_size, dim);
    for (int i = 0; i < vocab_size; i++) {
      for (int j = 0; j < dim; j++) {
        leftVectors.set(i, j, initializeValue());
        rightVectors.set(i, j, initializeValue());
      }
    }

    TDoubleArrayList wordsProbabsLeft = new TDoubleArrayList(vocab_size);
    TDoubleArrayList wordsProbabsRight = new TDoubleArrayList(vocab_size);
    final double X_sum = countWordsProbabs(wordsProbabsLeft, wordsProbabsRight);

    final Mx softMaxLeft = new VecBasedMx(leftVectors.rows(), leftVectors.columns());
    final Mx softMaxRight = new VecBasedMx(rightVectors.rows(), rightVectors.columns());
    VecTools.fill(softMaxLeft, 1.);
    VecTools.fill(softMaxRight, 1.);

    for (int iter = 0; iter < T(); iter++) {
      Interval.start();
      final ScoreCalculator scoreCalculator = new ScoreCalculator(vocab_size);
      int finalIter = iter;
      IntStream.range(0, vocab_size).parallel().forEach(i -> {
        final Vec left = leftVectors.row(i);
        final Vec softMaxL = softMaxLeft.row(i);
        cooc(i, (j, X_ij) -> {
          final Vec right = rightVectors.row(j);
          final Vec softMaxR = softMaxRight.row(j);
          final double asum = VecTools.multiply(left, right);
          //final double logMutualInfo = Math.log(X_ij) - Math.log(wordsProbabsLeft.get(i)) - Math.log(wordsProbabsRight.get(j)) + Math.log(X_sum);
          final double logMutualInfo = Math.log(X_ij) + Math.log(X_sum) - Math.log(wordsProbabsLeft.get(i)) - Math.log(wordsProbabsRight.get(j));
          //final double diff = asum - logMutualInfo;
          final double diff = (asum - (X_ij / X_sum) * logMutualInfo);
          //final double weight = weightingFunc(X_ij);
          final double weight = 1d;
          final double fdiff = step() * diff * weight;
          scoreCalculator.adjust(i, j, weight, 0.5 * weight * MathTools.sqr(diff));
          IntStream.range(0, dim).forEach(id -> {
            final double dL = fdiff * right.get(id);
            final double dR = fdiff * left.get(id);
            left.adjust(id, -dL / Math.sqrt(softMaxL.get(id)));
            right.adjust(id, -dR / Math.sqrt(softMaxR.get(id)));
            softMaxL.adjust(id, dL * dL);
            softMaxR.adjust(id, dR * dR);
          });
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
