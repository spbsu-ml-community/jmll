package com.expleague.ml.io;

import com.expleague.commons.func.types.ConversionPack;
import com.expleague.commons.math.Func;
import com.expleague.commons.math.Trans;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.ml.func.FuncEnsemble;
import com.expleague.commons.util.ArrayTools;
import com.expleague.commons.util.Pair;

import java.util.function.Function;

/**
 * User: qdeee
 * Date: 07.04.15
 */
public class FuncEnsembleConversionPack implements ConversionPack<FuncEnsemble, CharSequence> {
  public static class To extends EnsembleModelConversionPack.BaseTo<FuncEnsemble> {
    @Override
    public CharSequence convert(final FuncEnsemble from) {
      return convertModels(from);
    }
  }

  public static class From extends EnsembleModelConversionPack.BaseFrom<FuncEnsemble> {
    @Override
    public FuncEnsemble convert(final CharSequence from) {
      final Pair<Trans[], Vec> pair = convertModels(from);
      final Trans[] models = pair.getFirst();
      final Func[] funcModels = ArrayTools.map(models, Func.class, new Function<Trans, Func>() {
        @Override
        public Func apply(final Trans argument) {
          return (Func) argument;
        }
      });
      final Vec weights = pair.getSecond();
      return new FuncEnsemble(funcModels, weights);
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
