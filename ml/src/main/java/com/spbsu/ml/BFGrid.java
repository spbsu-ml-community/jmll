package com.spbsu.ml;

import com.spbsu.commons.func.Converter;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.io.BFGridStringConverter;

import java.util.Arrays;

/**
 * User: solar
 * Date: 09.11.12
 * Time: 17:56
 */
public class BFGrid {
  final BFRow[] rows;
  final BinaryFeature[] features;
  final int bfCount;
  final BFRow leastNonEmptyRow;

  public BFGrid(final BFRow[] rows) {
    this.rows = rows;
    for (final BFRow row : rows) {
      row.setOwner(this);
    }
    final BFRow lastRow = rows[rows.length - 1];
    bfCount = lastRow.bfStart + lastRow.borders.length;
    features = new BinaryFeature[bfCount];
    int rowIndex = 0;
    for (int i = 0; i < features.length; i++) {
      while (rowIndex < rows.length && i >= rows[rowIndex].bfEnd)
        rowIndex++;
      features[i] = rows[rowIndex].bf(i - rows[rowIndex].bfStart);
    }

    BFRow leastNonEmptyRow = null;
    for (int i = 0; i < rows.length; i++) {
      if (rows[i].size() > 0) {
        leastNonEmptyRow = leastNonEmptyRow != null ? (leastNonEmptyRow.size() > rows[i].size() ? rows[i] : leastNonEmptyRow) : rows[i];
      }
    }
    this.leastNonEmptyRow = leastNonEmptyRow;
  }

  public BFRow row(final int feature) {
    return feature < rows.length ? rows[feature] : new BFRow(this, bfCount, feature, new double[0]);
  }

  public BinaryFeature bf(final int bfIndex) {
    return features[bfIndex];
  }

  public void binarize(final Vec x, final byte[] folds) {
    for (int i = 0; i < x.dim(); i++) {
      folds[i] = (byte) rows[i].bin(x.get(i));
    }
  }

  public int size() {
    return features.length;
  }

  public int rows() {
    return rows.length;
  }

  public BFRow[] allRows() {
    return rows;
  }

  public BFRow nonEmptyRow() {
    return leastNonEmptyRow;
  }


  public static class BFRow {
    private BFGrid owner;
    public final int bfStart;
    public final int bfEnd;
    public final int origFIndex;
    public final double[] borders;
    public final BinaryFeature[] bfs;

    public BFRow(final BFGrid owner, final int bfStart, final int origFIndex, final double[] borders) {
      this(owner,bfStart,origFIndex,borders,new int[borders.length]);
    }
    public BFRow(final BFGrid owner, final int bfStart, final int origFIndex, final double[] borders, final int[] sizes) {
      this.owner = owner;
      this.bfStart = bfStart;
      this.bfEnd = bfStart + borders.length;
      this.origFIndex = origFIndex;
      this.borders = borders;
      bfs = new BinaryFeature[borders.length];
      for (int i = 0; i < borders.length; i++) {
        bfs[i] = new BinaryFeature(this, bfStart + i, origFIndex, i, borders[i],sizes[i]);
      }
    }

    public BFRow(final int bfStart, final int origFIndex, final double[] borders) {
      this(null, bfStart, origFIndex, borders);
    }

    public BFRow(final int bfStart, final int origFIndex, final double[] borders,final int[] sizes) {
      this(null, bfStart, origFIndex, borders,sizes);
    }

    public int bin(final double val) {
      int index = 0;
//      final int index = Arrays.binarySearch(borders, val);
//      return bfStart + (index >= 0 ? index : -index-1);
      while (index < borders.length && val > borders[index])
        index++;

      return index;
    }

    public BinaryFeature bf(final int index) {
      return bfs[index];
    }

    public double condition(final int border) {
      return borders[border];
    }

    public int size() {
      return bfEnd - bfStart;
    }

    public boolean empty() {
      return bfEnd == bfStart;
    }

    @Override
    public boolean equals(final Object o) {
      if (this == o) return true;
      if (!(o instanceof BFRow)) return false;

      final BFRow bfRow = (BFRow) o;

      return bfStart == bfRow.bfStart && origFIndex == bfRow.origFIndex && Arrays.equals(borders, bfRow.borders);

    }

    @Override
    public int hashCode() {
      int result = bfStart;
      result = 31 * result + origFIndex;
      result = 31 * result + Arrays.hashCode(borders);
      return result;
    }

    public BFGrid grid() {
      return owner;
    }

    private void setOwner(final BFGrid owner) {
      this.owner = owner;
    }
  }

  public static class BinaryFeature {
    private final BFRow bfRow;
    public final int bfIndex;
    public final int findex;
    public final int binNo;
    public final double condition;
    public final double size;

    public BinaryFeature(final BFRow bfRow, final int bfIndex, final int findex, final int binNo, final double condition) {
      this(bfRow, bfIndex, findex, binNo, condition, 0);
    }

    public BinaryFeature(final BFRow bfRow, final int bfIndex, final int findex, final int binNo, final double condition, int size) {
      this.bfRow = bfRow;
      this.bfIndex = bfIndex;
      this.findex = findex;
      this.binNo = binNo;
      this.condition = condition;
      this.size = size;
    }


    public boolean value(final byte[] folds) {
      return folds[findex] > binNo;
    }

    public boolean value(final Vec vec) {
      return vec.get(findex) > condition;
    }

    public BFRow row() {
      return bfRow;
    }

    @Override
    public boolean equals(final Object o) {
      if (this == o) return true;
      if (!(o instanceof BinaryFeature)) return false;

      final BinaryFeature that = (BinaryFeature) o;

      return bfIndex == that.bfIndex && bfRow.equals(that.bfRow);

    }

    @Override
    public int hashCode() {
      int result = bfRow.hashCode();
      result = 31 * result + bfIndex;
      return result;
    }

    @Override
    public String toString() {
      return String.format("f[%d] > %g", findex, condition);
    }
  }

  public static final Converter<BFGrid, CharSequence> CONVERTER = new BFGridStringConverter();

  @Override
  public boolean equals(final Object o) {
    if (this == o) return true;
    if (!(o instanceof BFGrid)) return false;

    final BFGrid bfGrid = (BFGrid) o;

    return Arrays.equals(rows, bfGrid.rows);

  }

  @Override
  public int hashCode() {
    return Arrays.hashCode(rows);
  }

  @Override
  public String toString() {
    return CONVERTER.convertTo(this).toString();
  }
}
