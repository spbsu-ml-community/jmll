package com.expleague.ml.data.tools;

import com.expleague.commons.seq.CharSeqTools;

import java.io.IOException;
import java.io.Reader;
import java.util.Arrays;

/**
 * Created by noxoomo on 15/10/2017.
 */
public class CatBoostPoolDescription {

  enum ColumnType {
    Num,
    Categ,
    Target,
    Auxiliary,
    DocId,
    QueryId,
    Weight;

    static boolean isFactorColumn(ColumnType type) {
      return type == Num || type == Categ;
    }
  }

  private ColumnType[] columnTypes;
  private char delimiter = '\t';
  private boolean headerColumnFlag = false;


  public CatBoostPoolDescription(int columnCount) {
    columnTypes = new ColumnType[columnCount];
    Arrays.fill(columnTypes, ColumnType.Num);
    columnTypes[0] = ColumnType.Target;
  }

  ColumnType columnType(int columnIndex) {
    return columnTypes[columnIndex];
  }

  public char getDelimiter() {
    return delimiter;
  }

  public boolean hasHeaderColumn() {
    return headerColumnFlag;
  }

  public int columnCount() {
    return columnTypes.length;
  }

  public int factorCount() {
    int total = 0;
    for (ColumnType type : columnTypes) {
      if (ColumnType.isFactorColumn(type)) {
        total++;
      }
    }
    return total;
  }

  private void validate() {
    int targetCount = 0;
    int weightCount = 0;
    int featureCount = 0;
    for (ColumnType type : columnTypes) {
      switch (type) {
        case Target: {
          ++targetCount;
          break;
        }
        case Num:
        case Categ: {
          ++featureCount;
          break;
        }
        case Weight: {
          ++weightCount;
          break;
        }
      }
    }
    if ((targetCount != 1) || (weightCount > 1) || (featureCount == 0)) {
      throw new RuntimeException("Wrong pool description format");
    }
  }

  public static class DescriptionBuilder {
    private final CatBoostPoolDescription description;
    private int columnCount;

    public DescriptionBuilder(int columnCount) {
      this.columnCount = columnCount;
      this.description = new CatBoostPoolDescription(columnCount);
    }

    public DescriptionBuilder(final Reader poolInput,
                              final char delimiter) {
      this.columnCount = DataTools.getLineCount(poolInput, delimiter);
      this.description = new CatBoostPoolDescription(columnCount);
      description.delimiter = delimiter;
    }

    public DescriptionBuilder loadColumnDescription(final Reader input) throws IOException {
      description.columnTypes = new ColumnType[columnCount];
      Arrays.fill(description.columnTypes, ColumnType.Num);
      CharSeqTools.processLines(input, arg -> {
        final CharSequence[] parts = CharSeqTools.split(arg, '\t');

        final int index = CharSeqTools.parseInt(parts[0]);
        description.columnTypes[index] = ColumnType.valueOf(parts[1].toString());
      });
      return this;
    }

    public DescriptionBuilder setDelimiter(char delimiter) {
      description.delimiter = delimiter;
      return this;
    }

    public DescriptionBuilder setHasHeaderColumnFlag(boolean headerColumn) {
      description.headerColumnFlag = headerColumn;
      return this;
    }

    public CatBoostPoolDescription description() {
      description.validate();
      return description;
    }
  }
}

