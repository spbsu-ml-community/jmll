package com.spbsu.ml;

import com.spbsu.commons.func.Converter;
import com.spbsu.commons.math.vectors.Vec;
import gnu.trove.TDoubleArrayList;

import java.io.IOException;
import java.io.LineNumberReader;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;

/**
 * User: solar
 * Date: 09.11.12
 * Time: 17:56
 */
public class BFGrid {
  final BFRow[] rows;
  final BinaryFeature[] features;
  final int bfCount;

  public BFGrid(BFRow[] rows) {
    this.rows = rows;
    final BFRow lastRow = rows[rows.length - 1];
    bfCount = lastRow.bfStart + lastRow.borders.length;
    features = new BinaryFeature[bfCount];
    int rowIndex = 0;
    for (int i = 0; i < features.length; i++) {
      while (rowIndex < rows.length && i >= rows[rowIndex].bfEnd)
        rowIndex++;
      features[i] = rows[rowIndex].bf(i - rows[rowIndex].bfStart);
    }
  }

  public BFRow row(int feature) {
    return feature < rows.length ? rows[feature] : new BFRow(bfCount, feature, new double[0]);
  }

  public BinaryFeature bf(int bfIndex) {
    return features[bfIndex];
  }

  public void binarize(Vec x, byte[] folds) {
    for (int i = 0; i < x.dim(); i++) {
      folds[i] = (byte)rows[i].bin(x.get(i));
    }
  }

  public int size() {
    return features.length;
  }

  public int rows() {
    return rows.length;
  }


  public static class BFRow {
    public final int bfStart;
    public final int bfEnd;
    public final int origFIndex;
    public final double[] borders;
    public final BinaryFeature[] bfs;

    public BFRow(int bfStart, int origFIndex, double[] borders) {
      this.bfStart = bfStart;
      this.bfEnd = bfStart + borders.length;
      this.origFIndex = origFIndex;
      this.borders = borders;
      bfs = new BinaryFeature[borders.length];
      for (int i = 0; i < borders.length; i++) {
        bfs[i] = new BinaryFeature(this, bfStart + i, origFIndex, i, borders[i]);
      }
    }

    public int bin(double val) {
      int index = 0;
//      final int index = Arrays.binarySearch(borders, val);
//      return bfStart + (index >= 0 ? index : -index-1);
      while (index < borders.length && val > borders[index])
        index++;

      return index;
    }

    public BinaryFeature bf(int index) {
      return bfs[index];
    }

    public double condition(int border) {
      return borders[border];
    }

    public int size() {
      return bfEnd - bfStart;
    }

    public boolean empty() {
      return bfEnd == bfStart;
    }
  }

  public static class BinaryFeature {
    private final BFRow bfRow;
    public final int bfIndex;
    public final int findex;
    public final int binNo;
    public final double condition;

    public BinaryFeature(BFRow bfRow, int bfIndex, int findex, int binNo, double condition) {
      this.bfRow = bfRow;
      this.bfIndex = bfIndex;
      this.findex = findex;
      this.binNo = binNo;
      this.condition = condition;
    }

    public boolean value(byte[] folds) {
      return folds[findex] > binNo;
    }

    public boolean value(Vec vec) {
      return vec.get(findex) > condition;
    }

    public BFRow row() {
      return bfRow;
    }
  }

  public static final Converter<BFGrid, String> CONVERTER = new Converter<BFGrid, String>() {
    @Override
    public BFGrid convertFrom(String source) {
      final List<BFRow> rows = new ArrayList<BFRow>(1000);
      final LineNumberReader reader = new LineNumberReader(new StringReader(source));
      String line;
      try {
        final TDoubleArrayList borders = new TDoubleArrayList();
        int bfIndex = 0;
        int lineIndex = 0;
        while ((line = reader.readLine()) != null) {
          final StringTokenizer tok = new StringTokenizer(line, " \t");
          borders.clear();
          while (tok.hasMoreElements()) {
            borders.add(Double.parseDouble(tok.nextToken()));
          }
          rows.add(new BFRow(bfIndex, lineIndex, borders.toNativeArray()));
          bfIndex += borders.size();
          lineIndex++;
        }
      }
      catch (IOException ioe) { //skip
      }
      return new BFGrid(rows.toArray(new BFRow[rows.size()]));
    }

    @Override
    public String convertTo(BFGrid grid) {
      StringBuilder builder = new StringBuilder();
      for (int rowIndex = 0; rowIndex < grid.rows.length; rowIndex++) {
        final BFRow row = grid.row(rowIndex);
        final int rowBfCount = row.borders.length;
        for (int border = 0; border < rowBfCount; border++) {
          builder.append(border > 0 ? "\t" : "");
          builder.append(row.condition(border));
        }
        builder.append("\n");
      }
      return builder.toString();
    }
  };
}
