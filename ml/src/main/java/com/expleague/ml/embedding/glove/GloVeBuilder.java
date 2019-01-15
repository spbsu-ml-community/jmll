package com.expleague.ml.embedding.glove;

import com.expleague.commons.math.MathTools;
import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecIterator;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.seq.CharSeq;
import com.expleague.commons.util.logging.Interval;
import com.expleague.ml.embedding.Embedding;
import com.expleague.ml.embedding.impl.EmbeddingBuilderBase;
import com.expleague.ml.embedding.impl.EmbeddingImpl;
import gnu.trove.iterator.TIntDoubleIterator;

import java.util.HashMap;
import java.util.Map;
import java.util.stream.IntStream;

public class GloVeBuilder extends EmbeddingBuilderBase {
  private double xMax = 10;
  private double alpha = 0.75;
  private int dim = 50;

  public GloVeBuilder xMax(int xMax) {
    this.xMax = xMax;
    return this;
  }

  public GloVeBuilder alpha(double alpha) {
    this.alpha = alpha;
    return this;
  }

  public GloVeBuilder dim(int dim) {
    this.dim = dim;
    return this;
  }

  private double weightingFunc(double x) {
    return x < xMax ? Math.pow((x / xMax), alpha) : 1;
  }

  @Override
  public Embedding<CharSeq> fit() {
    final int vocab_size = dict().size();
    final Mx leftVectors = new VecBasedMx(vocab_size, dim + 1);
    final Mx rightVectors = new VecBasedMx(vocab_size, dim + 1);
    final Vec biasLeft = new ArrayVec(vocab_size);
    final Vec biasRight = new ArrayVec(vocab_size);
    for (int i = 0; i < vocab_size; i++) {
      biasRight.set(i, initializeValue());
      biasLeft.set(i, initializeValue());
      for (int j = 0; j < dim; j++) {
        leftVectors.set(i, j, initializeValue());
        rightVectors.set(i, j, initializeValue());
      }
    }

    final Mx softMaxLeft = new VecBasedMx(leftVectors.rows(), leftVectors.columns());
    final Mx softMaxRight = new VecBasedMx(rightVectors.rows(), rightVectors.columns());
    final Vec softBiasLeft = new ArrayVec(biasLeft.dim());
    final Vec softBiasRight = new ArrayVec(biasRight.dim());
    VecTools.fill(softMaxLeft, 1.);
    VecTools.fill(softMaxRight, 1.);
    VecTools.fill(softBiasLeft, 1.);
    VecTools.fill(softBiasRight, 1.);

    for (int iter = 0; iter < T(); iter++) {
      Interval.start();
      final ScoreCalculator scoreCalculator = new ScoreCalculator(dim);
      IntStream.range(0, vocab_size).parallel().forEach(i -> {
        final Vec left = leftVectors.row(i);
        final Vec softMaxL = softMaxLeft.row(i);
        cooc(i, (j, X_ij) -> {
          final Vec right = rightVectors.row(j);
          final Vec softMaxR = softMaxRight.row(j);
          final double asum = VecTools.multiply(left, right);
          final double diff = biasLeft.get(i) + biasRight.get(j) + asum - Math.log(X_ij);
          final double weight = weightingFunc(X_ij);
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

          biasLeft.adjust(i, -fdiff / Math.sqrt(softBiasLeft.get(i)));
          biasRight.adjust(j, -fdiff / Math.sqrt(softBiasRight.get(j)));
          softBiasLeft.adjust(i, MathTools.sqr(fdiff));
          softBiasRight.adjust(j, MathTools.sqr(fdiff));
        });
      });

      Interval.stopAndPrint("Iteration " + iter + ", score: " + scoreCalculator.gloveScore() + ", count: " + scoreCalculator.count());
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
