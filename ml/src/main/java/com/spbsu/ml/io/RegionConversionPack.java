package com.spbsu.ml.io;

import com.spbsu.commons.func.types.ConversionPack;
import com.spbsu.commons.func.types.TypeConverter;
import com.spbsu.commons.seq.CharSeqReader;
import com.spbsu.commons.seq.CharSeqTools;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.GridEnabled;
import com.spbsu.ml.models.Region;
import gnu.trove.list.array.TLongArrayList;

import java.io.IOException;
import java.io.LineNumberReader;
import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.text.MessageFormat;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * User: noxoomo
 * Date: 11.11.14
 */

public class RegionConversionPack implements ConversionPack<Region, CharSequence> {
  private static final MessageFormat FEATURE_LINE_PATTERN = new MessageFormat("feature: {0, number}, bin: {1, number}, ge: {2, number,#.#####}, mask : {3, number}", Locale.US);

  static {
    final DecimalFormat format = new DecimalFormat();
    format.setDecimalSeparatorAlwaysShown(false);
    format.setGroupingUsed(false);
    format.setDecimalFormatSymbols(new DecimalFormatSymbols(Locale.US));
    format.setDecimalSeparatorAlwaysShown(true);
    format.setMaximumFractionDigits(5);
    format.setParseIntegerOnly(false);
    format.setParseIntegerOnly(false);

    FEATURE_LINE_PATTERN.setFormat(2, format);
  }

  public static class To implements TypeConverter<Region, CharSequence> {
    @Override
    public CharSequence convert(final Region region) {
      final StringBuilder result = new StringBuilder();
      final BFGrid.BinaryFeature[] features = region.features();
      final boolean[] masks = region.masks();
      for (int i = 0; i < features.length; ++i) {
        result.append(FEATURE_LINE_PATTERN.format(new Object[]{features[i].findex, features[i].binNo, features[i].condition, masks[i] ? 1 : 0})).append("\n");
      }
      result.append(region.inside)
              .append(":")
              .append(region.outside)
              .append(":")
              .append(region.maxFailed)
              .append(":")
              .append(region.basedOn)
              .append(":")
              .append(region.score).append("\n");
      return result;
    }
  }

  public static class From implements GridEnabled, TypeConverter<CharSequence, Region> {
    private BFGrid grid;

    @Override
    public BFGrid getGrid() {
      return grid;
    }

    @Override
    public void setGrid(final BFGrid grid) {
      this.grid = grid;
    }

    @Override
    public Region convert(final CharSequence source) {
      if (grid == null)
        throw new RuntimeException("Grid must be setup for serialization of oblivious trees, use SerializationRepository.customize!");
      String line;
      final LineNumberReader lnr = new LineNumberReader(new CharSeqReader(source));
      final List<BFGrid.BinaryFeature> splits = new ArrayList<BFGrid.BinaryFeature>(10);
      final TLongArrayList mask = new TLongArrayList();
      try {
        while ((line = lnr.readLine()) != null) {
          if (line.startsWith("feature")) {
            final Object[] parts = FEATURE_LINE_PATTERN.parse(line);
            final BFGrid.BinaryFeature bf = grid.row(((Long) parts[0]).intValue()).bf(((Long) parts[1]).intValue());
            splits.add(bf);
            if (Math.abs(bf.condition - ((Number) parts[2]).doubleValue()) > 1e-4)
              throw new RuntimeException("Inconsistent grid set, conditions do not match! Grid: " + bf.condition + " Found: " + parts[2]);
            mask.add((Long) parts[3]);
          } else break;
        }

        final CharSequence[] pattern2ValueBased = CharSeqTools.split(line, ':');
        final double inside = Double.parseDouble(pattern2ValueBased[0].toString());
        final double outside = Double.parseDouble(pattern2ValueBased[1].toString());
        final int maxFailed = Integer.parseInt(pattern2ValueBased[2].toString());
        final int basedOn = Integer.parseInt(pattern2ValueBased[3].toString());
        final double score = Double.parseDouble(pattern2ValueBased[4].toString());
        final boolean[] masks = new boolean[mask.size()];
        for (int i = 0; i < masks.length; ++i)
          masks[i] = mask.get(i) == 1;
        return new Region(splits, masks, inside, outside, basedOn, score, maxFailed);
      } catch (
              IOException e
              )

      {
        throw new RuntimeException(e);
      } catch (
              ParseException e
              )

      {
        throw new RuntimeException("Invalid region format", e);
      }
    }
  }

  @Override
  public Class<? extends TypeConverter<Region, CharSequence>> to() {
    return To.class;
  }

  @Override
  public Class<? extends TypeConverter<CharSequence, Region>> from() {
    return From.class;
  }
}
