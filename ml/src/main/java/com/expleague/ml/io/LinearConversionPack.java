package com.expleague.ml.io;

import com.expleague.commons.func.types.ConversionDependant;
import com.expleague.commons.func.types.ConversionPack;
import com.expleague.commons.func.types.ConversionRepository;
import com.expleague.commons.func.types.TypeConverter;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.ml.func.Linear;

/**
 * User: solar
 * Date: 26.01.19
 */

public class LinearConversionPack implements ConversionPack<Linear, CharSequence> {
  public static class To implements TypeConverter<Linear, CharSequence>, ConversionDependant {
    private ConversionRepository conv;
    @Override
    public void setConversionRepository(ConversionRepository repository) {
      this.conv = repository;
    }
    @Override
    public CharSequence convert(final Linear region) {
      return conv.convert(region.weights, CharSequence.class);
    }
  }

  public static class From implements ConversionDependant, TypeConverter<CharSequence, Linear> {
    private ConversionRepository conv;
    @Override
    public void setConversionRepository(ConversionRepository repository) {
      this.conv = repository;
    }

    @Override
    public Linear convert(final CharSequence source) {
      return new Linear(conv.convert(source, Vec.class));
    }
  }

  @Override
  public Class<? extends TypeConverter<Linear, CharSequence>> to() {
    return To.class;
  }

  @Override
  public Class<? extends TypeConverter<CharSequence, Linear>> from() {
    return From.class;
  }
}
