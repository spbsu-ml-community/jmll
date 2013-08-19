package com.spbsu.ml.io;

import com.spbsu.commons.func.types.ConversionPack;
import com.spbsu.commons.func.types.TypeConverter;
import com.spbsu.commons.text.CharSequenceReader;
import com.spbsu.commons.text.CharSequenceTools;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.GridEnabled;
import com.spbsu.ml.models.ObliviousTree;

import java.io.IOException;
import java.io.LineNumberReader;
import java.text.MessageFormat;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * User: solar
 * Date: 12.08.13
 * Time: 13:05
 */
public class ObliviousTreeConversionPack implements ConversionPack<ObliviousTree, CharSequence> {
  private static final MessageFormat FEATURE_LINE_PATTERN = new MessageFormat("feature: {0, number}, bin: {1, number}, condition_value: {2, number,#.#####}", Locale.US);

  public static class To implements TypeConverter <ObliviousTree, CharSequence> {
    @Override
    public CharSequence convert(ObliviousTree ot) {
      StringBuilder result = new StringBuilder();
      for (BFGrid.BinaryFeature feature : ot.getFeatures()) {
        result.append(FEATURE_LINE_PATTERN.format(new Object[]{feature.findex, feature.binNo, feature.condition}))
              .append("\n");
      }
      int leafsCount = 1 << ot.getFeatures().size();
      for (int i = 0; i < leafsCount; i++) {
        if (ot.values()[i] != 0.) {
          result.append(Integer.toBinaryString(i))
                  .append(":").append(ot.values()[i])
                  .append(":").append(ot.based()[i])
                  .append(" ");
        }
      }
      result = result.delete(result.length() - 1, result.length());
      result.append('\n');

      return result;
    }
  }

  public static class From implements GridEnabled, TypeConverter<CharSequence, ObliviousTree> {
    private BFGrid grid;

    public BFGrid getGrid() {
      return grid;
    }

    @Override
    public void setGrid(BFGrid grid) {
      this.grid = grid;
    }

    @Override
    public ObliviousTree convert(CharSequence source) {
      if (grid == null)
        throw new RuntimeException("Grid must be setup for serialization of oblivious trees, use SerializationRepository.customize!");
      String line;
      LineNumberReader lnr = new LineNumberReader(new CharSequenceReader(source));
      List<BFGrid.BinaryFeature> splits = new ArrayList<BFGrid.BinaryFeature>(10);
      try {
        while ((line = lnr.readLine()) != null) {
          if (line.startsWith("feature")) {
            final Object[] parts = FEATURE_LINE_PATTERN.parse(line);
            BFGrid.BinaryFeature bf = grid.row(((Long)parts[0]).intValue()).bf(((Long)parts[1]).intValue());
            splits.add(bf);
            if (Math.abs(bf.condition - (Double)parts[2]) > 1e-4)
              throw new RuntimeException("Inconsistent grid set, conditions does not match:");
          }
          else break;
        }
        double[] values = new double[1 << splits.size()];
        double[] based = new double[1 << splits.size()];
        CharSequence[] valuesStr = CharSequenceTools.split(line, ' ');
        for (CharSequence value : valuesStr) {
          final CharSequence[] pattern2ValueBased = CharSequenceTools.split(value, ':');
          final int leafIndex = Integer.parseInt(pattern2ValueBased[0].toString(), 2);
          values[leafIndex] = Double.parseDouble(pattern2ValueBased[1].toString());
          based[leafIndex] = Double.parseDouble(pattern2ValueBased[2].toString());
        }
        return new ObliviousTree(splits, values, based);
      } catch (IOException e) {
        throw new RuntimeException(e);
      } catch (ParseException e) {
        throw new RuntimeException("Invalid oblivious tree format", e);
      }
    }
  }

  @Override
  public Class<? extends TypeConverter<ObliviousTree, CharSequence>> to() {
    return To.class;
  }

  @Override
  public Class<? extends TypeConverter<CharSequence, ObliviousTree>> from() {
    return From.class;
  }
}
