package com.expleague.ml.io;

import com.expleague.commons.func.types.ConversionDependant;
import com.expleague.commons.func.types.ConversionPack;
import com.expleague.commons.func.types.ConversionRepository;
import com.expleague.commons.func.types.TypeConverter;
import com.expleague.commons.math.Func;
import com.expleague.commons.math.Trans;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.seq.CharSeqTools;
import com.expleague.commons.util.Pair;
import com.expleague.ml.func.Ensemble;
import com.expleague.ml.func.ScaledVectorFunc;
import com.expleague.ml.func.TransJoin;

import java.util.StringTokenizer;

/**
 * User: solar
 * Date: 12.08.13
 * Time: 17:16
 */
@SuppressWarnings("unused")
public class ScaledFuncConversionPack implements ConversionPack<ScaledVectorFunc, CharSequence> {
  public static class To implements TypeConverter<ScaledVectorFunc, CharSequence>, ConversionDependant {
    private ConversionRepository repository;

    @Override
    public void setConversionRepository(final ConversionRepository repository) {
      this.repository = repository;
    }

    @Override
    public CharSequence convert(ScaledVectorFunc from) {
      final StringBuilder builder = new StringBuilder();
      builder.append(repository.convert(from.weights, CharSequence.class)).append('\n');
      builder.append(from.function.getClass().getName()).append('\n');
      builder.append(repository.convert(from.function, CharSequence.class));
      return builder;
    }
  }

  public static class From implements TypeConverter<CharSequence, ScaledVectorFunc>, ConversionDependant {

    private ConversionRepository repository;

    @Override
    public void setConversionRepository(final ConversionRepository repository) {
      this.repository = repository;
    }

    @Override
    public ScaledVectorFunc convert(CharSequence from) {
      if (from.toString().indexOf('\r') >= 0)
        from = from.toString().replace("\r", ""); // fix windows newlines created by GIT

      CharSequence[] parts = new CharSequence[3];
      CharSeqTools.split(from, '\n', parts);

      try {
        final Vec weight = repository.convert(parts[0], Vec.class);
        //noinspection unchecked
        final Class<? extends Func> elementClass = (Class<? extends Func>) Class.forName(parts[1].toString());
        final Func model = repository.convert(parts[2], elementClass);
        return new ScaledVectorFunc(model, weight);
      }
      catch (ClassNotFoundException e) {
        throw new RuntimeException("Element class not found!", e);
      }
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
