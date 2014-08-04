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


  static <T extends BinarySaver> int writeArray(DataOutputStream stream, T[] vector) throws IOException {
    ByteBuffer buffer = ByteBuffer.allocate(4);
    buffer.order(byteOrder);
    buffer.putInt(vector.length);
    stream.write(buffer.array());
    for (BinarySaver elem : vector) {
      elem.add(stream);
    }
    return 0;
  }


  static <T extends BinarySaver> int writeList(DataOutputStream stream, List<T> list) throws IOException {
    ByteBuffer buffer = ByteBuffer.allocate(4);
    buffer.order(byteOrder);
    buffer.putInt(list.size());
    stream.write(buffer.array());
    for (BinarySaver elem : list) {
      elem.add(stream);
    }
    return 0;
  }

  static int writeList(DataOutputStream stream, TDoubleArrayList values) throws IOException {
    ByteBuffer buffer = ByteBuffer.allocate(4 + 8 * values.size());
    buffer.order(byteOrder);
    buffer.putInt(values.size());
    for (int i = 0; i < values.size(); ++i) {
      double value = (values.get(i));
      buffer.putDouble(value);
    }
    stream.write(buffer.array());
    return 0;
  }


  static int writeList(DataOutputStream stream, TFloatArrayList values) throws IOException {
    ByteBuffer buffer = ByteBuffer.allocate(4 + 4 * values.size());
    buffer.order(byteOrder);
    buffer.putInt(values.size());
    for (int i = 0; i < values.size(); ++i) {
      float value = (values.get(i));
      buffer.putFloat(value);
    }
    stream.write(buffer.array());
    return 0;
  }

  static ByteOrder byteOrder = ByteOrder.LITTLE_ENDIAN;

  public static int writeArray(DataOutputStream stream, int[] values) throws IOException {
    ByteBuffer buffer = ByteBuffer.allocate(4 + 4 * values.length);
    buffer.order(byteOrder);
    buffer.putInt(values.length);
    for (int value : values) {
      buffer.putInt(value);
    }
    stream.write(buffer.array());
    return 0;
  }

  public static int writeArray(DataOutputStream stream, double[] values) throws IOException {
    ByteBuffer buffer = ByteBuffer.allocate(4 + 8 * values.length);
    buffer.order(byteOrder);
    buffer.putInt(values.length);
    for (double value : values) {
      buffer.putDouble(value);
    }
    stream.write(buffer.array());
    return 0;
  }


  public static int write2DArray(DataOutputStream stream, double[][] values) throws IOException {
    ByteBuffer buffer = ByteBuffer.allocate(4);
    buffer.order(byteOrder);
    buffer.putInt(values.length);
    stream.write(buffer.array());
    for (double[] value : values) {
      writeArray(stream, value);
    }
    return 0;
  }
}


class FullMatrixClassifierInfo extends BinarySaver {

  BinaryFeatureStat[] binFeatures;
  List<TreeStat> trees;
  TFloatArrayList classValues;

  public FullMatrixClassifierInfo(int gridSize) {
    binFeatures = new BinaryFeatureStat[gridSize];
    trees = new ArrayList<>();
    classValues = new TFloatArrayList();
  }

  public void toFile(File file) {
    try {
      FileOutputStream fileOut = new FileOutputStream(file);
      DataOutputStream out = new DataOutputStream(fileOut);
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
  public boolean add(DataOutputStream stream) throws IOException {
    BinarySaver.writeArray(stream, binFeatures);
    BinarySaver.writeList(stream, trees);
    BinarySaver.writeList(stream, classValues);
    return true;
  }
}

class BinaryFeatureStat extends BinarySaver {
  int origFIndex;
  float border;

  public BinaryFeatureStat(int origFIndex, double border) {
    this.origFIndex = origFIndex;
    this.border = (float) border;
  }

  @Override
  public boolean add(DataOutputStream stream) throws IOException {
    ByteBuffer buffer = ByteBuffer.allocate(4 + 4);
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

  TreeStat(int conditions[], double[][] values) {
    this.conditions = conditions;
    this.values = values;
  }

  @Override
  public boolean add(DataOutputStream stream) throws IOException {
    BinarySaver.writeArray(stream, conditions);
    BinarySaver.write2DArray(stream, values);
    return true;
  }
}

