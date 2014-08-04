package com.spbsu.ml.io;

import com.spbsu.commons.func.Converter;
import com.spbsu.ml.DynamicGrid.Interface.DynamicGrid;
import com.spbsu.ml.DynamicGrid.Interface.DynamicRow;

/**
 * User: solar
 * Date: 12.08.13
 * Time: 21:15
 */
public class BFDynamicGridStringConverter implements Converter<DynamicGrid, CharSequence> {
//  @Override
//  public DynamicGrid convertFrom(CharSequence source) {
//    final List<BFGrid.BFRow> rows = new ArrayList<BFGrid.BFRow>(1000);
//    final LineNumberReader reader = new LineNumberReader(new CharSeqReader(source));
//    String line;
//    try {
//      final TDoubleArrayList borders = new TDoubleArrayList();
//      int bfIndex = 0;
//      int lineIndex = 0;
//      while ((line = reader.readLine()) != null) {
//        final StringTokenizer tok = new StringTokenizer(line, " \t");
//        borders.clear();
//        while (tok.hasMoreElements()) {
//          borders.add(Double.parseDouble(tok.nextToken()));
//        }
//        rows.add(new BFGrid.BFRow(bfIndex, lineIndex, borders.toArray()));
//        bfIndex += borders.size();
//        lineIndex++;
//      }
//    }
//    catch (IOException ioe) { //skip
//    }
//    return new BFDynamicGrid(rows.toArray(new BFGrid.BFRow[rows.size()]));
//  }

  @Override
  public CharSequence convertTo(DynamicGrid grid) {
    StringBuilder builder = new StringBuilder();
    for (int rowIndex = 0; rowIndex < grid.rows(); rowIndex++) {
      final DynamicRow row = grid.row(rowIndex);
      final int rowBfCount = row.size();
      for (int border = 0; border < rowBfCount; border++) {
        builder.append(border > 0 ? "\t" : "");
        builder.append(row.bf(border).condition());
      }
      builder.append("\n");
    }
    return builder;
  }

  @Override
  public DynamicGrid convertFrom(CharSequence source) {
    return null;
  }
}
