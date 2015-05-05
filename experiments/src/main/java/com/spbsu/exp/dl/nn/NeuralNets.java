package com.spbsu.exp.dl.nn;

import org.jetbrains.annotations.NotNull;

import com.spbsu.exp.dl.functions.unary.DoubleArrayUnaryFunction;

import com.spbsu.commons.math.vectors.impl.mx.ColMajorArrayMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.exp.dl.functions.unary.IdenticalFA;
import com.spbsu.exp.dl.functions.unary.SigmoidFA;
import com.spbsu.exp.dl.utils.DataUtils;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Random;

import static com.spbsu.ml.cuda.JCublasHelper.*;

/**
 * jmll
 * ksen
 * 23.November.2014 at 15:47
 */
public class NeuralNets {

  public ColMajorArrayMx[] weights;
  public ColMajorArrayMx[] activations;
  public DoubleArrayUnaryFunction rectifier;
  public DoubleArrayUnaryFunction outputRectifier;
  public int batchSize;

  public NeuralNets(
      final @NotNull int[] layersDimensions,
      final int batchSize,
      final @NotNull DoubleArrayUnaryFunction rectifier,
      final @NotNull DoubleArrayUnaryFunction outputRectifier,
      final @NotNull InitMethod initMethod
  ) {
    final int length = layersDimensions.length;

    weights = new ColMajorArrayMx[length - 1];
    activations = new ColMajorArrayMx[length];

    for (int i = 0; i < length - 1; i++) {
      weights[i] = new ColMajorArrayMx(layersDimensions[i + 1], layersDimensions[i] + 1);
      activations[i] = new ColMajorArrayMx(layersDimensions[i], batchSize);
    }
    activations[length - 1] = new ColMajorArrayMx(layersDimensions[length - 1], batchSize);

    this.rectifier = rectifier;
    this.outputRectifier = outputRectifier;
    this.batchSize = batchSize;
    init(initMethod);
  }

  public NeuralNets(
      final @NotNull ColMajorArrayMx[] weights,
      final @NotNull ColMajorArrayMx[] activations,
      final @NotNull DoubleArrayUnaryFunction rectifier,
      final @NotNull DoubleArrayUnaryFunction outputRectifier
  ) {
    this.weights = weights;
    this.activations = activations;
    this.rectifier = rectifier;
    this.outputRectifier = outputRectifier;
  }

  public NeuralNets(final @NotNull String path2model) {
    read(path2model);
  }

  private void init(final @NotNull InitMethod initMethod) {
    switch (initMethod) {
      case RANDOM_SMALL : {
        final Random random = new Random(1);
        final ColMajorArrayMx[] W = weights;

        for (int i = 0; i < W.length; i++) {
          final ColMajorArrayMx wights = W[i];

          final int rows = wights.rows();
          final int columns = wights.columns();
          final float scale = 8.f * (float) Math.sqrt(6. / (rows + columns - 1.));
          for (int j = 0; j < rows; j++) {
            for (int k = 0; k < columns; k++) {
              wights.set(j, k, (random.nextFloat() - 0.5f) * scale);
            }
          }
        }
        break;
      }
      case DO_NOTHING : {
        break;
      }
    }
  }

  public ArrayVec forward(final @NotNull ArrayVec x) {
    ArrayVec input = DataUtils.extendAsBottom(x, 1.f);
    ArrayVec output;

    final int last = weights.length;
    for (int i = 1; i < last; i++) {
      output = rectifier.f(mult(weights[i - 1], input));
      input = DataUtils.extendAsBottom(output, 1.f);
    }
    return outputRectifier.f(mult(weights[last - 1], input));
  }

  public ColMajorArrayMx batchForward(final @NotNull ColMajorArrayMx X) {
    final ArrayVec once = DataUtils.once(X.columns());
    activations[0] = DataUtils.extendAsBottomRow(X, once);

    final int last = weights.length;
    for (int i = 1; i < last; i++) {
      activations[i] = rectifier.f(mult(weights[i - 1], activations[i - 1]));

      //todo(ksenon): dropout, sparsity

      activations[i] = DataUtils.extendAsBottomRow(activations[i], once);
    }
    activations[last] = outputRectifier.f(mult(weights[last - 1], activations[last - 1]));

    return activations[last];
  }

  public void read(final @NotNull String path) {
    try (
        final RandomAccessFile raf = new RandomAccessFile(path, "rw");
        final FileChannel fc = raf.getChannel()
    ) {
      final MappedByteBuffer byteBuffer = fc.map(FileChannel.MapMode.READ_WRITE, 0, fc.size());

      byteBuffer.getInt();
      rectifier = new SigmoidFA();
      byteBuffer.getInt();
      outputRectifier = new IdenticalFA();
      weights = new ColMajorArrayMx[byteBuffer.getInt()];

      for (int i = 0; i < weights.length; i++) {
        final ColMajorArrayMx W = new ColMajorArrayMx(byteBuffer.getInt(), byteBuffer.getInt());

        for (int row = 0; row < W.rows(); row++) {
          for (int column = 0; column < W.columns(); column++) {
            W.set(row, column, byteBuffer.getFloat());
          }
        }
        weights[i] = W;
      }
    }
    catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  public void write(final @NotNull String path) {
    long size = 3 + 2 * weights.length;
    for (int i = 0; i < weights.length; i++) {
      size += weights[i].rows() * weights[i].columns();
    }
    try (
        final RandomAccessFile raf = new RandomAccessFile(path, "rw");
        final FileChannel fc = raf.getChannel()
    ) {
      final MappedByteBuffer byteBuffer = fc.map(FileChannel.MapMode.READ_WRITE, 0, 4 * size);

      byteBuffer.putInt(0);
      byteBuffer.putInt(1);
      byteBuffer.putInt(weights.length);

      for (int i = 0; i < weights.length; i++) {
        final ColMajorArrayMx W = weights[i];
        byteBuffer.putInt(W.rows());
        byteBuffer.putInt(W.columns());

        for (int row = 0; row < W.rows(); row++) {
          for (int column = 0; column < W.columns(); column++) {
            byteBuffer.putDouble(W.get(row, column));
          }
        }
      }
    }
    catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

}