package com.expleague.ml.io;

import com.expleague.commons.func.Computable;
import com.expleague.commons.func.types.ConversionPack;
import com.expleague.commons.util.ArrayTools;
import com.expleague.commons.math.Func;
import com.expleague.commons.math.Trans;
import com.expleague.ml.models.multiclass.JoinedProbsModel;

/**
 * User: qdeee
 * Date: 31.07.14
 */
public class JoinedProbsModelConversionPack implements ConversionPack<JoinedProbsModel, CharSequence> {
  public static class To extends TransJoinConversionPack.BaseTo<JoinedProbsModel> {
    @Override
    public CharSequence convert(final JoinedProbsModel from) {
      return convertModels(from);
    }
  }

  public static class From extends TransJoinConversionPack.BaseFrom<JoinedProbsModel> {
    @Override
    public JoinedProbsModel convert(final CharSequence from) {
      final Func[] dirs = ArrayTools.map(convertModels(from), Func.class, new Computable<Trans, Func>() {
        @Override
        public Func compute(final Trans argument) {
          return (Func) argument;
        }
      });
      return new JoinedProbsModel(dirs);
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
