package com.spbsu.ml.io;

import com.spbsu.commons.func.types.ConversionDependant;
import com.spbsu.commons.func.types.ConversionPack;
import com.spbsu.commons.func.types.ConversionRepository;
import com.spbsu.commons.func.types.TypeConverter;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.seq.CharSeqTools;
import com.spbsu.commons.util.Pair;
import com.spbsu.commons.math.Trans;
import com.spbsu.ml.func.Ensemble;

import java.util.StringTokenizer;

/**
 * User: solar
 * Date: 12.08.13
 * Time: 17:16
 */
public class EnsembleModelConversionPack implements ConversionPack<Ensemble, CharSequence> {
  public static abstract class BaseTo<F extends Ensemble> implements TypeConverter<F, CharSequence>, ConversionDependant {
    private ConversionRepository repository;

    @Override
    public void setConversionRepository(final ConversionRepository repository) {
      this.repository = repository;
    }

    protected CharSequence convertModels(final F from) {
      final StringBuilder builder = new StringBuilder();
      builder.append(from.size());
      builder.append("\n\n");
      for (int i = 0; i < from.size(); i++) {
        final Trans model = from.models[i];
        builder.append(from.models[i].getClass().getCanonicalName()).append(" ");
        builder.append(from.weights.get(i)).append("\n");
        builder.append(repository.convert(model, CharSequence.class));
        builder.append("\n\n");
      }
      builder.delete(builder.length() - 1, builder.length());
      return builder;
    }
  }

  public static class To extends BaseTo<Ensemble> {
    @Override
    public CharSequence convert(final Ensemble from) {
      return convertModels(from);
    }
  }

  public abstract static class BaseFrom<T extends Ensemble> implements TypeConverter<CharSequence, T>, ConversionDependant {

    private ConversionRepository repository;

    @Override
    public void setConversionRepository(final ConversionRepository repository) {
      this.repository = repository;
    }
    protected Pair<Trans[], Vec> convertModels(CharSequence from) {
      if (from.toString().indexOf('\r') >= 0)
        from = from.toString().replace("\r", ""); // fix windows newlines created by GIT

      final CharSequence[] elements = CharSeqTools.split(from, "\n\n");
      final Trans[] models;
      final double[] weights;

      try {
        final int count = Integer.parseInt(elements[0].toString());
        models = new Trans[count];
        weights = new double[count];
        for (int i = 0; i < count; i++) {
          final CharSequence[] lines = CharSeqTools.split(elements[i + 1], "\n");
          final StringTokenizer tok = new StringTokenizer(lines[0].toString(), " ");
          final Class<? extends Trans> elementClass = (Class<? extends Trans>) Class.forName(tok.nextToken());
          weights[i] = Double.parseDouble(tok.nextToken());
          models[i] = repository.convert(elements[i + 1].subSequence(lines[0].length() + 1, elements[i + 1].length()), elementClass);
        }
      } catch (ClassNotFoundException e) {
        throw new RuntimeException("Element class not found!", e);
      }
      return Pair.create(models, (Vec) new ArrayVec(weights));
    }
  }

  public static class From extends BaseFrom<Ensemble> {
    @Override
    public Ensemble convert(final CharSequence from) {
      final Pair<Trans[], Vec> pair = convertModels(from);
      final Trans[] models = pair.getFirst();
      final Vec weights = pair.getSecond();
      return new Ensemble(models, weights);
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
