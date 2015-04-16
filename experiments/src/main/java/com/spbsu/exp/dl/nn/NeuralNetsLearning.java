package com.spbsu.exp.dl.nn;

import org.jetbrains.annotations.NotNull;

import com.spbsu.exp.dl.functions.binary.HadamardFA;

import com.spbsu.commons.math.vectors.impl.mx.ColMajorArrayMx;
import com.spbsu.exp.dl.utils.DataUtils;
import gnu.trove.list.TIntList;
import gnu.trove.list.array.TIntArrayList;

import static com.spbsu.exp.dl.utils.DataUtils.*;
import static com.spbsu.ml.cuda.JCublasHelper.*;

/**
 * jmll
 * ksen
 * 23.November.2014 at 15:52
 */
public class NeuralNetsLearning {

  private NeuralNets nn;
  private float alpha;
//  private float momentum;
//  private float scalingAlpha;
//  private float weightsPenalty;
//  private float nonSparsityPenalty;
//  private float sparsityTarget;
//  private float dropoutLevel;
  private int epochsNumber;
  private int batchSize;
  private final HadamardFA FHadamard = new HadamardFA();

  public NeuralNetsLearning(
      final @NotNull NeuralNets nn,
      final float alpha,
      final int epochsNumber
  ) {
    this.nn = nn;
    this.alpha = alpha;
//    this.momentum = momentum;
//    this.scalingAlpha = scalingAlpha;
//    this.weightsPenalty = weightsPenalty;
//    this.nonSparsityPenalty = nonSparsityPenalty;
//    this.sparsityTarget = sparsityTarget;
//    this.dropoutLevel = dropoutLevel;
    this.epochsNumber = epochsNumber;
    this.batchSize = nn.batchSize;
  }

  public void batchLearn(final @NotNull ColMajorArrayMx X, final @NotNull ColMajorArrayMx Y) {
    final int examplesNumber = X.columns();
    final int batchesNumber = examplesNumber / batchSize;
    final int lastLayerIndex = nn.weights.length;

    for (int i = 0; i < epochsNumber; i++) {
      final TIntArrayList examplesIndexes = DataUtils.randomPermutations(examplesNumber);

      for (int j = 0; j < batchesNumber; j++) {
        final TIntList indexes = examplesIndexes.subList(j * batchSize, (j + 1) * batchSize);
        final ColMajorArrayMx batchX = X.getColumnsRange(indexes);
        final ColMajorArrayMx batchY = Y.getColumnsRange(indexes);

        final ColMajorArrayMx output = nn.batchForward(batchX);
        final ColMajorArrayMx error = subtr(batchY, output);

        final ColMajorArrayMx[] D = backPropagation(error);
        updateWeights(D);
      }
      System.out.println("Epoch " + i);
    }
  }

  private ColMajorArrayMx[] backPropagation(final ColMajorArrayMx error) {
    final int lastLayerIndex = nn.weights.length;
    final ColMajorArrayMx[] D = new ColMajorArrayMx[lastLayerIndex + 1];
    for (int i = 0; i < lastLayerIndex + 1; i++) {
      final ColMajorArrayMx activation = nn.activations[i];
      D[i] = new ColMajorArrayMx(activation.rows(), activation.columns());
    }

    D[lastLayerIndex] = FHadamard.f(scale(-1., error), nn.outputRectifier.df(nn.activations[lastLayerIndex]));

    for (int i = lastLayerIndex - 1; i > 0; i--) {
      final ColMajorArrayMx dA = nn.rectifier.df(nn.activations[i]);

      //todo(ksen): non sparsity penalty

      if (i + 1 == lastLayerIndex) {
        D[i] = FHadamard.f(mult(nn.weights[i], true, D[i + 1], false), dA); //todo(ksen): sparsity error
      }
      else {
        D[i] = FHadamard.f(mult(nn.weights[i], true, contractBottomRow(D[i + 1]), false), dA); //todo(ksen): sparsity error
      }

      //todo(ksen): dropout
    }
    for (int i = 0; i < lastLayerIndex; i++) {
      if (i + 1 == lastLayerIndex) {
        D[i] = scale(1. / D[i + 1].columns(), mult(D[i + 1], false, nn.activations[i], true));
      }
      else {
        D[i] = scale(1. / D[i + 1].columns(), mult(contractBottomRow(D[i + 1]), false, nn.activations[i], true));
      }
    }
    return D;
  }

  private void updateWeights(final ColMajorArrayMx[] D) {
    final int weightsNumber = nn.weights.length;
    for (int i = 0; i < weightsNumber; i++) {
      //todo(ksen): L2 penalty

      final ColMajorArrayMx dW = scale(alpha, D[i]);

      //todo(ksen): momentum

      nn.weights[i] = subtr(nn.weights[i], dW);
    }
  }

}
