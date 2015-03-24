package com.spbsu.ml.data.tools;

import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.list.array.TFloatArrayList;

import java.io.DataOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by noxoomo on 31/07/14.
 */
public abstract class BinarySaver {
  public abstract boolean add(DataOutputStream stream) throws IOException;


  static <T extends BinarySaver> int writeArray(final DataOutputStream stream, final T[] vector) throws IOException {
    final ByteBuffer buffer = ByteBuffer.allocate(4);
    buffer.order(byteOrder);
    buffer.putInt(vector.length);
    stream.write(buffer.array());
    for (final BinarySaver elem : vector) {
      elem.add(stream);
    }
    return 0;
  }


  static <T extends BinarySaver> int writeList(final DataOutputStream stream, final List<T> list) throws IOException {
    final ByteBuffer buffer = ByteBuffer.allocate(4);
    buffer.order(byteOrder);
    buffer.putInt(list.size());
    stream.write(buffer.array());
    for (final BinarySaver elem : list) {
      elem.add(stream);
    }
    return 0;
  }

  static int writeList(final DataOutputStream stream, final TDoubleArrayList values) throws IOException {
    final ByteBuffer buffer = ByteBuffer.allocate(4 + 8 * values.size());
    buffer.order(byteOrder);
    buffer.putInt(values.size());
    for (int i = 0; i < values.size(); ++i) {
      final double value = (values.get(i));
      buffer.putDouble(value);
    }
    stream.write(buffer.array());
    return 0;
  }


  static int writeList(final DataOutputStream stream, final TFloatArrayList values) throws IOException {
    final ByteBuffer buffer = ByteBuffer.allocate(4 + 4 * values.size());
    buffer.order(byteOrder);
    buffer.putInt(values.size());
    for (int i = 0; i < values.size(); ++i) {
      final float value = (values.get(i));
      buffer.putFloat(value);
    }
    stream.write(buffer.array());
    return 0;
  }

  static ByteOrder byteOrder = ByteOrder.LITTLE_ENDIAN;

  public static int writeArray(final DataOutputStream stream, final int[] values) throws IOException {
    final ByteBuffer buffer = ByteBuffer.allocate(4 + 4 * values.length);
    buffer.order(byteOrder);
    buffer.putInt(values.length);
    for (final int value : values) {
      buffer.putInt(value);
    }
    stream.write(buffer.array());
    return 0;
  }

  public static int writeArray(final DataOutputStream stream, final double[] values) throws IOException {
    final ByteBuffer buffer = ByteBuffer.allocate(4 + 8 * values.length);
    buffer.order(byteOrder);
    buffer.putInt(values.length);
    for (final double value : values) {
      buffer.putDouble(value);
    }
    stream.write(buffer.array());
    return 0;
  }


  public static int write2DArray(final DataOutputStream stream, final double[][] values) throws IOException {
    final ByteBuffer buffer = ByteBuffer.allocate(4);
    buffer.order(byteOrder);
    buffer.putInt(values.length);
    stream.write(buffer.array());
    for (final double[] value : values) {
      writeArray(stream, value);
    }
    return 0;
  }
}


class FullMatrixClassifierInfo extends BinarySaver {

  BinaryFeatureStat[] binFeatures;
  List<TreeStat> trees;
  TFloatArrayList classValues;

  public FullMatrixClassifierInfo(final int gridSize) {
    binFeatures = new BinaryFeatureStat[gridSize];
    trees = new ArrayList<>();
    classValues = new TFloatArrayList();
  }

  public void toFile(final File file) {
    try {
      final FileOutputStream fileOut = new FileOutputStream(file);
      final DataOutputStream out = new DataOutputStream(fileOut);
      add(out);
      out.flush();
      fileOut.flush();
      fileOut.close();
      out.close();
    } catch (IOException e) {
      System.err.println("Can't write model");
    }

  }

  @Override
  public boolean add(final DataOutputStream stream) throws IOException {
    BinarySaver.writeArray(stream, binFeatures);
    BinarySaver.writeList(stream, trees);
    BinarySaver.writeList(stream, classValues);
    return true;
  }
}

class BinaryFeatureStat extends BinarySaver {
  int origFIndex;
  float border;

  public BinaryFeatureStat(final int origFIndex, final double border) {
    this.origFIndex = origFIndex;
    this.border = (float) border;
  }

  @Override
  public boolean add(final DataOutputStream stream) throws IOException {
    final ByteBuffer buffer = ByteBuffer.allocate(4 + 4);
    buffer.order(BinarySaver.byteOrder);
    buffer.putInt(origFIndex);
    buffer.putFloat(border);
    stream.write(buffer.array());
    return true;
  }
}


class TreeStat extends BinarySaver {

  final int conditions[];
  final double[][] values;

  TreeStat(final int[] conditions, final double[][] values) {
    this.conditions = conditions;
    this.values = values;
  }

  @Override
  public boolean add(final DataOutputStream stream) throws IOException {
    BinarySaver.writeArray(stream, conditions);
    BinarySaver.write2DArray(stream, values);
    return true;
  }
}

