package com.spbsu.exp.dl.dnn;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import gnu.trove.list.TIntList;
import gnu.trove.list.array.TIntArrayList;

import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

/**
 * jmll
 *
 * @author ksenon
 */
public class Solver {

  public FullyConnectedNet net;

  public InitMethod initMethod;

  public int batchSize;
  public int epochsNumber;

  public double learningRate;

  public boolean debug;

  public void solve(final Mx X, final Mx Y) {
    final int examplesNumber = X.rows();
    final int batchesNumber = examplesNumber / batchSize;

    for (int i = 0; i < epochsNumber; i++) {
      final TIntArrayList examplesIndexes = randomPermutations(examplesNumber);

      for (int j = 0; j < batchesNumber; j++) {
        final TIntList indexes = examplesIndexes.subList(j * batchSize, (j + 1) * batchSize);
        final Mx batch_X = getRows(indexes, X);
        final Mx batch_Y = getRows(indexes, Y);

        net.input = batch_X;
        net.forward();

        final Mx error = VecTools.subtract(batch_Y, net.output);

        net.backward(error);
        updateWeights();
      }
    }
  }

  public static TIntArrayList randomPermutations(final int size) {
    final TIntArrayList list = new TIntArrayList(size);

    for (int i = 0; i < size; i++) {
      list.add(i);
    }
    list.shuffle(ThreadLocalRandom.current());

    return list;
  }

  public Mx getRows(final TIntList indexes, final Mx X) {
    final Mx batch_X = new VecBasedMx(indexes.size(), X.columns());

    for (int i = 0; i < indexes.size(); i++) {
      final Vec row_X = X.row(indexes.get(i));

      for (int j = 0; j < row_X.dim(); j++) {
        batch_X.set(i, j, row_X.get(j));
      }
    }
    return batch_X;
  }

  private void updateWeights() {
    final Layer[] layers = net.layers;
    for (final Layer layer : layers) {
      for (int i = 0; i < layer.difference.dim(); i++) {
        layer.weights.set(i, layer.weights.get(i) - learningRate * layer.difference.get(i));
      }
    }
  }

  public void init() {
    switch (initMethod) {
      case RANDOM_OPTIMAL : {
        final Random random = new Random();
        for (final Layer layer : net.layers) {
          final Mx weights = layer.weights;
          final int rows = weights.rows();
          final int columns = weights.columns();

          final double scale = 8. * Math.sqrt(6. / (rows + columns - 1.));
          for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
              weights.set(i, j, scale * (random.nextDouble() - 0.5));
            }
          }
        }
        break;
      }
    }
  }

}
