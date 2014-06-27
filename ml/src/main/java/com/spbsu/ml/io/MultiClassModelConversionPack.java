package com.spbsu.ml.io;

import com.spbsu.commons.func.Computable;
import com.spbsu.commons.func.types.ConversionPack;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.ml.Func;
import com.spbsu.ml.Trans;
import com.spbsu.ml.models.MultiClassModel;

/**
 * User: starlight
 * Date: 17.04.14
 */
public class MultiClassModelConversionPack implements ConversionPack<MultiClassModel, CharSequence> {
  public static class To extends TransJoinConversionPack.BaseTo<MultiClassModel> {
    @Override
    public CharSequence convert(final MultiClassModel from) {
      return convertModels(from);
    }
  }

  public static class From extends TransJoinConversionPack.BaseFrom<MultiClassModel> {
    @Override
    public MultiClassModel convert(final CharSequence from) {
      //TODO starlight: remove casting
      final Func[] dirs = ArrayTools.map(convertModels(from), Func.class, new Computable<Trans, Func>() {
        @Override
        public Func compute(Trans argument) {
          return (Func) argument;
        }
      });
      return new MultiClassModel(dirs);
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
