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
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static com.expleague.commons.math.vectors.VecTools.append;

public class GloVeBuilder extends CoocBasedBuilder {
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
    final Mx leftVectors = new VecBasedMx(vocab_size, dim);
    final Mx rightVectors = new VecBasedMx(vocab_size, dim);
    //final Map<CharSeq, Vec> trigramVectors = new HashMap<>();
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

    TDoubleArrayList wordsProbabsLeft = new TDoubleArrayList(vocab_size);
    TDoubleArrayList wordsProbabsRight = new TDoubleArrayList(vocab_size);
    final double X_sum = countWordsProbabs(wordsProbabsLeft, wordsProbabsRight);

    final Mx softMaxLeft = new VecBasedMx(leftVectors.rows(), leftVectors.columns());
    final Mx softMaxRight = new VecBasedMx(rightVectors.rows(), rightVectors.columns());
    final Vec softBiasLeft = new ArrayVec(biasLeft.dim());
    final Vec softBiasRight = new ArrayVec(biasRight.dim());
    VecTools.fill(softMaxLeft, 1.);
    VecTools.fill(softMaxRight, 1.);
    VecTools.fill(softBiasLeft, 1.);
    VecTools.fill(softBiasRight, 1.);


    Mx leftBiasPaths = new VecBasedMx(vocab_size, T() + 1);
    Mx rightBiasPaths = new VecBasedMx(vocab_size, T() + 1);
    for (int i = 0; i < vocab_size; i++) {
      leftBiasPaths.set(i, 0, biasLeft.get(i));
      rightBiasPaths.set(i, 0, biasRight.get(i));
    }

    /*for (int i = 0; i < 50; i++) {
      System.out.println("Word: " + dict().get(i));
      System.out.println("Freq left:  " + wordsProbabsLeft.get(i) + "  Freq right:  " + wordsProbabsRight.get(i));
      System.out.println("probabs left: " + wordsProbabsLeft.get(i) / X_sum + "  Log(probabs right): " + Math.log(wordsProbabsRight.get(i) / X_sum));
    };*/

    for (int iter = 0; iter < T(); iter++) {
      Interval.start();
      final ScoreCalculator scoreCalculator = new ScoreCalculator(vocab_size);
      int finalIter = iter;
      IntStream.range(0, vocab_size).parallel().forEach(i -> {
        final CharSeq word_i = dict().get(i);
        final Vec left = leftVectors.row(i);
        final Vec softMaxL = softMaxLeft.row(i);
        cooc(i, (j, X_ij) -> {
          final CharSeq word_j = dict().get(j);
          final Vec right = rightVectors.row(j);
          final Vec softMaxR = softMaxRight.row(j);
          final double asum = VecTools.multiply(left, right);
          final double logMutualInfo = Math.log(X_ij) - Math.log(wordsProbabsLeft.get(i)) - Math.log(wordsProbabsRight.get(j)) + Math.log(X_sum);
          //final double diff = biasLeft.get(i) + biasRight.get(j) + asum - Math.log(X_ij);
          final double diff = asum - logMutualInfo;
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

          /*biasLeft.adjust(i, -fdiff / Math.sqrt(softBiasLeft.get(i)));
          biasRight.adjust(j, -fdiff / Math.sqrt(softBiasRight.get(j)));
          softBiasLeft.adjust(i, MathTools.sqr(fdiff));
          softBiasRight.adjust(j, MathTools.sqr(fdiff));*/

        });
      });
      IntStream.range(0, vocab_size).parallel().forEach(i -> {
        leftBiasPaths.set(i, finalIter + 1, biasLeft.get(i));
        rightBiasPaths.set(i, finalIter + 1, biasRight.get(i));
      });
      Interval.stopAndPrint("Iteration " + iter + ", score: " + scoreCalculator.gloveScore());
    }

    final Map<CharSeq, Vec> mapping = new HashMap<>();
    for (int i = 0; i < dict().size(); i++) {
      final CharSeq word = dict().get(i);
      mapping.put(word, VecTools.sum(leftVectors.row(i), rightVectors.row(i)));
    }

    System.out.println("FINISHED TRAINING");
    System.out.println("------biases------");
    /*IntStream.range(0, vocab_size).parallel().forEach(i -> {
      cooc(i, (j, X_ij) -> {
        final double biased = biasLeft.get(i) + biasRight.get(j) - Math.log(X_ij);
        double probabs = Math.log(wordsProbabsLeft.get(i)) + Math.log(wordsProbabsRight.get(j)) - Math.log(X_ij) - Math.log(X_sum);
        //probabs = probabs * weightingFunc(X_ij) / X_sum;
        System.out.println("Biased part " + biased + ", probabs part " + probabs);
      });
    });*/

    /*for (int i = 0; i < 50; i++) {
      System.out.println("Word: " + dict().get(i));
      System.out.println("Freq left:  " + wordsProbabsLeft.get(i) + "  Freq right:  " + wordsProbabsRight.get(i));
      System.out.println("Log(probabs left): " + Math.log(wordsProbabsLeft.get(i) / X_sum) + "  Log(probabs right): " + Math.log(wordsProbabsRight.get(i) / X_sum));
    };
    /*for (int i = 0; i < 50; i++) {
      CharSeq word = dict().get(i);
      System.out.println("Word: " + word + "\nleft biases: " + leftBiasPaths.row(i).toString() + "\nright biases: "  + rightBiasPaths.row(i).toString() + "\n");
    }*/
    System.out.println("------------------");

    return new EmbeddingImpl<>(mapping);
  }

  private synchronized Vec trigram(CharSeq sub, Map<CharSeq, Vec> trigramVectors) {
    return trigramVectors.computeIfAbsent(sub, characters -> new ArrayVec(dim));
  }

  private double initializeValue() {
    return (Math.random() - 0.5) / dim;
  }
}
