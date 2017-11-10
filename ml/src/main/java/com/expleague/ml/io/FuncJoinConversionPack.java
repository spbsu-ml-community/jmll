package com.expleague.ml.io;

import com.expleague.commons.func.types.ConversionPack;
import com.expleague.commons.util.ArrayTools;
import com.expleague.commons.math.Func;
import com.expleague.ml.func.FuncJoin;


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
      final Func[] dirs = ArrayTools.map(convertModels(from), Func.class, argument -> (Func) argument);
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
