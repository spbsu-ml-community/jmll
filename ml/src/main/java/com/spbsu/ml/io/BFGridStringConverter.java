package com.spbsu.ml.io;

import com.spbsu.commons.func.Converter;
import com.spbsu.commons.seq.CharSeqReader;
import com.spbsu.ml.BFGrid;
import gnu.trove.list.array.TDoubleArrayList;
import java.io.IOException;
import java.io.LineNumberReader;
import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;

/**
* User: solar
* Date: 12.08.13
* Time: 21:15
*/
public class BFGridStringConverter implements Converter<BFGrid, CharSequence> {
  @Override
  public BFGrid convertFrom(final CharSequence source) {
    final List<BFGrid.BFRow> rows = new ArrayList<BFGrid.BFRow>(1000);
    final LineNumberReader reader = new LineNumberReader(new CharSeqReader(source));
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
        rows.add(new BFGrid.BFRow(bfIndex, lineIndex, borders.toArray()));
        bfIndex += borders.size();
        lineIndex++;
      }
    }
    catch (IOException ioe) { //skip
    }
    return new BFGrid(rows.toArray(new BFGrid.BFRow[rows.size()]));
  }

  @Override
  public CharSequence convertTo(final BFGrid grid) {
    final StringBuilder builder = new StringBuilder();
    for (int rowIndex = 0; rowIndex < grid.rows(); rowIndex++) {
      final BFGrid.BFRow row = grid.row(rowIndex);
      final int rowBfCount = row.borders.length;
      for (int border = 0; border < rowBfCount; border++) {
        builder.append(border > 0 ? "\t" : "");
        builder.append(row.condition(border));
      }
      builder.append("\n");
    }
    return builder;
  }
}
