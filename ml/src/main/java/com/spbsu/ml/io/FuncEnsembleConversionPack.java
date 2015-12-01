package com.spbsu.ml.io;

import com.spbsu.commons.func.Computable;
import com.spbsu.commons.func.types.ConversionPack;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.commons.util.Pair;
import com.spbsu.commons.math.Func;
import com.spbsu.commons.math.Trans;
import com.spbsu.ml.func.FuncEnsemble;

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
      final Func[] funcModels = ArrayTools.map(models, Func.class, new Computable<Trans, Func>() {
        @Override
        public Func compute(final Trans argument) {
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
