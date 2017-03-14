package com.spbsu.ml.io;

import com.spbsu.commons.func.Computable;
import com.spbsu.commons.func.types.ConversionPack;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.commons.math.Func;
import com.spbsu.commons.math.Trans;
import com.spbsu.ml.func.FuncJoin;

/**
 * User: qdeee
 * Date: 31.07.14
 */
public class FuncJoinConversionPack implements ConversionPack<FuncJoin, CharSequence> {
  public static class To extends TransJoinConversionPack.BaseTo<FuncJoin> {
    @Override
    public CharSequence convert(final FuncJoin from) {
      return convertModels(from);
    }
  }

  public static class From extends TransJoinConversionPack.BaseFrom<FuncJoin> {
    @Override
    public FuncJoin convert(final CharSequence from) {
      final Func[] dirs = ArrayTools.map(convertModels(from), Func.class, new Computable<Trans, Func>() {
        @Override
        public Func compute(final Trans argument) {
          return (Func) argument;
        }
      });
      return new FuncJoin(dirs);
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
