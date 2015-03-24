package com.spbsu.ml.io;

import com.spbsu.commons.func.converters.Vec2StringConverter;
import com.spbsu.commons.func.types.ConversionPack;
import com.spbsu.commons.func.types.TypeConverter;
import com.spbsu.commons.math.io.Vec2CharSequenceConverter;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.seq.CharSeqTools;
import com.spbsu.ml.models.FMModel;

/**
 * User: qdeee
 * Date: 31.03.14
 */
public class FMModelConversionPack implements ConversionPack<FMModel, CharSequence> {
  public static class To implements TypeConverter<FMModel, CharSequence> {
    @Override
    public CharSequence convert(final FMModel from) {
      final Vec2StringConverter vec2StringConverter = new Vec2StringConverter();
      final StringBuilder builder = new StringBuilder();
      builder.append(String.valueOf(from.getW0()));
      builder.append("\n");
      builder.append(vec2StringConverter.convertTo(from.getW()));
      builder.append("\n");
      builder.append(vec2StringConverter.convertTo(from.getV()));
      return builder;
    }
  }

  public static class From implements TypeConverter<CharSequence, FMModel> {
    @Override
    public FMModel convert(final CharSequence from) {
      final Vec2CharSequenceConverter vec2StringConverter = new Vec2CharSequenceConverter();
      final CharSequence[] lines = CharSeqTools.split(from, '\n');
      final double w0 = Double.valueOf(lines[0].toString());
      final Vec w = vec2StringConverter.convertFrom(lines[1].toString());
      final Vec vecV = vec2StringConverter.convertFrom(lines[2].toString());
      final Mx V = new VecBasedMx(w.dim(), vecV);
      return new FMModel(V, w, w0);
    }
  }


  @Override
  public Class<To> to() {
    return To.class;
  }

  @Override
  public Class<From> from() {
    return From.class;
  }
}
