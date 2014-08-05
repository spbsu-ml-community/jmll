package com.spbsu.ml.io;

import com.spbsu.commons.func.types.ConversionPack;
import com.spbsu.commons.func.types.TypeConverter;
import com.spbsu.commons.seq.CharSeqReader;
import com.spbsu.commons.seq.CharSeqTools;
import com.spbsu.ml.DynamicGridEnabled;
import com.spbsu.ml.dynamicGridFix.interfacesFix.BinaryFeature;
import com.spbsu.ml.dynamicGridFix.interfacesFix.DynamicGrid;
import com.spbsu.ml.dynamicGridFix.modelsFix.ObliviousTreeDynamicBin;

import java.io.IOException;
import java.io.LineNumberReader;
import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.text.MessageFormat;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public class ObliviousTreeDynamicBinConversionPack implements ConversionPack<ObliviousTreeDynamicBin, CharSequence> {
  private static final MessageFormat FEATURE_LINE_PATTERN = new MessageFormat("feature: {0, number}, bin: {1, number}, ge: {2, number,#.#####}", Locale.US);

  static {
    DecimalFormat format = new DecimalFormat();
    format.setDecimalSeparatorAlwaysShown(false);
    format.setGroupingUsed(false);
    format.setDecimalFormatSymbols(new DecimalFormatSymbols(Locale.US));
    format.setDecimalSeparatorAlwaysShown(true);
    format.setMaximumFractionDigits(5);
    format.setParseIntegerOnly(false);
    format.setParseIntegerOnly(false);
    FEATURE_LINE_PATTERN.setFormat(2, format);
  }

  public static class To implements TypeConverter<ObliviousTreeDynamicBin, CharSequence> {
    @Override
    public CharSequence convert(ObliviousTreeDynamicBin ot) {
      StringBuilder result = new StringBuilder();
      for (BinaryFeature feature : ot.features()) {
        result.append(FEATURE_LINE_PATTERN.format(new Object[]{feature.fIndex(), feature.binNo(), feature.condition()}))
                .append("\n");
      }
      int leafsCount = 1 << ot.features().length;
      for (int i = 0; i < leafsCount; i++) {
        if (ot.values()[i] != 0.) {
          result.append(Integer.toBinaryString(i))
                  .append(":").append(ot.values()[i])
                  .append(":").append(0)
                  .append(" ");
        }
      }
      result = result.delete(result.length() - 1, result.length());
      return result;
    }
  }

  public static class From implements DynamicGridEnabled, TypeConverter<CharSequence, ObliviousTreeDynamicBin> {
    private DynamicGrid grid;

    public DynamicGrid getGrid() {
      return grid;
    }

    @Override
    public void setGrid(DynamicGrid grid) {
      this.grid = grid;
    }

    @Override
    public ObliviousTreeDynamicBin convert(CharSequence source) {
      if (grid == null)
        throw new RuntimeException("DynamicGrid must be setup for serialization of oblivious trees with dynamicGrid, use SerializationRepository.customize!");
      String line;
      LineNumberReader lnr = new LineNumberReader(new CharSeqReader(source));
      List<BinaryFeature> splits = new ArrayList<BinaryFeature>(10);
      try {
        while ((line = lnr.readLine()) != null) {
          if (line.startsWith("feature")) {
            final Object[] parts = FEATURE_LINE_PATTERN.parse(line);
            BinaryFeature bf = grid.row(((Long) parts[0]).intValue()).bf(((Long) parts[1]).intValue());
            splits.add(bf);
            if (Math.abs(bf.condition() - ((Number) parts[2]).doubleValue()) > 1e-4)
              throw new RuntimeException("Inconsistent grid set, conditions do not match! Grid: " + bf.condition() + " Found: " + parts[2]);
          } else break;
        }
        double[] values = new double[1 << splits.size()];
        CharSequence[] valuesStr = CharSeqTools.split(line, ' ');
        for (CharSequence value : valuesStr) {
          final CharSequence[] pattern2ValueBased = CharSeqTools.split(value, ':');
          final int leafIndex = Integer.parseInt(pattern2ValueBased[0].toString(), 2);
          values[leafIndex] = Double.parseDouble(pattern2ValueBased[1].toString());
        }
        return new ObliviousTreeDynamicBin(splits, values);
      } catch (IOException e) {
        throw new RuntimeException(e);
      } catch (ParseException e) {
        throw new RuntimeException("Invalid oblivious tree format", e);
      }
    }
  }

  @Override
  public Class<? extends TypeConverter<ObliviousTreeDynamicBin, CharSequence>> to() {
    return To.class;
  }

  @Override
  public Class<? extends TypeConverter<CharSequence, ObliviousTreeDynamicBin>> from() {
    return From.class;
  }
}
