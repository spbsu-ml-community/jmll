package com.spbsu.ml.io;

import com.spbsu.commons.func.types.ConversionDependant;
import com.spbsu.commons.func.types.ConversionPack;
import com.spbsu.commons.func.types.ConversionRepository;
import com.spbsu.commons.func.types.TypeConverter;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.seq.CharSeqTools;
import com.spbsu.ml.Trans;
import com.spbsu.ml.func.Ensemble;

import java.util.StringTokenizer;

/**
 * User: solar
 * Date: 12.08.13
 * Time: 17:16
 */
public class EnsembleModelConversionPack implements ConversionPack<Ensemble, CharSequence> {
  public static class To implements TypeConverter<Ensemble, CharSequence>, ConversionDependant {
    private ConversionRepository repository;

    @Override
    public CharSequence convert(Ensemble from) {
      StringBuilder builder = new StringBuilder();
      builder.append(from.size());
      builder.append("\n\n");
      for (int i = 0; i < from.size(); i++) {
        Trans model = from.models[i];
        builder.append(from.models[i].getClass().getCanonicalName()).append(" ");
        builder.append(from.weights.get(i)).append("\n");
        builder.append(repository.convert(model, CharSequence.class));
        builder.append("\n\n");
      }
      builder.delete(builder.length() - 1, builder.length());
      return builder;
    }

    @Override
    public void setConversionRepository(ConversionRepository repository) {
      this.repository = repository;
    }
  }

  public static class From implements TypeConverter<CharSequence, Ensemble>, ConversionDependant {
    private ConversionRepository repository;

    @Override
    public Ensemble convert(CharSequence from) {
      CharSequence[] elements = CharSeqTools.split(from, "\n\n");
      Trans[] models;
      double[] weights;

      try {
        int count = Integer.parseInt(elements[0].toString());
        models = new Trans[count];
        weights = new double[count];
        for (int i = 0; i < count; i++) {
          final CharSequence[] lines = CharSeqTools.split(elements[i + 1], "\n");
          StringTokenizer tok = new StringTokenizer(lines[0].toString(), " ");
          Class<? extends Trans> elementClass = (Class<? extends Trans>) Class.forName(tok.nextToken());
          weights[i] = Double.parseDouble(tok.nextToken());
          models[i] = repository.convert(elements[i + 1].subSequence(lines[0].length() + 1, elements[i + 1].length()), elementClass);
        }
      } catch (ClassNotFoundException e) {
        throw new RuntimeException("Element class not found!", e);
      }
      return new Ensemble(models, new ArrayVec(weights));
    }

    @Override
    public void setConversionRepository(ConversionRepository repository) {
      this.repository = repository;
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
