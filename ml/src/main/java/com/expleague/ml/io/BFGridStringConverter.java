package com.expleague.ml.io;

import com.expleague.commons.func.Converter;
import com.expleague.commons.seq.CharSeqReader;
import com.expleague.ml.impl.BFGridImpl;
import com.expleague.ml.BFGrid;
import com.expleague.ml.impl.BFRowImpl;
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
    final List<BFRowImpl> rows = new ArrayList<>(1000);
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
        rows.add(new BFRowImpl(bfIndex, lineIndex, borders.toArray()));
        bfIndex += borders.size();
        lineIndex++;
      }
    }
    catch (IOException ioe) { //skip
    }
    return new BFGridImpl(rows.toArray(new BFRowImpl[rows.size()]));
  }

  @Override
  public CharSequence convertTo(final BFGrid grid) {
    final StringBuilder builder = new StringBuilder();
    for (int rowIndex = 0; rowIndex < grid.rows(); rowIndex++) {
      final BFGrid.Row row = grid.row(rowIndex);
      final int rowBfCount = row.size();
      for (int border = 0; border < rowBfCount; border++) {
        builder.append(border > 0 ? "\t" : "");
        builder.append(row.condition(border));
      }
      builder.append("\n");
    }
    return builder;
  }
}
