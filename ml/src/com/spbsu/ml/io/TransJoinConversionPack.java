package com.spbsu.ml.io;

import com.spbsu.commons.func.types.ConversionDependant;
import com.spbsu.commons.func.types.ConversionPack;
import com.spbsu.commons.func.types.ConversionRepository;
import com.spbsu.commons.func.types.TypeConverter;
import com.spbsu.commons.text.CharSequenceTools;
import com.spbsu.ml.Trans;
import com.spbsu.ml.func.TransJoin;

/**
 * User: solar
 * Date: 12.08.13
 * Time: 17:16
 */
public class TransJoinConversionPack implements ConversionPack<TransJoin, CharSequence> {

  public static abstract class BaseTo<F extends TransJoin> implements TypeConverter<F, CharSequence>, ConversionDependant {
    private ConversionRepository repository;

    @Override
    public void setConversionRepository(final ConversionRepository repository) {
      this.repository = repository;
    }

    protected CharSequence convertModels(F from) {
      StringBuilder builder = new StringBuilder();
      builder.append("{").append(from.dirs.length).append(",\n");
      for (int i = 0; i < from.dirs.length; i++) {
        builder.append("{");
        Trans model = from.dirs[i];
        builder.append(from.dirs[i].getClass().getCanonicalName()).append(",\n");
        builder.append("{");
        builder.append(repository.convert(model, CharSequence.class));
        builder.append("}");
        builder.append("},\n");
      }
      builder.delete(builder.length() - 1, builder.length());
      builder.append("}");
      return builder;
    }
  }

  public static class To extends BaseTo<TransJoin> {
    @Override
    public CharSequence convert(TransJoin from) {
      return convertModels(from);
    }
  }

  public static abstract class BaseFrom<T extends TransJoin> implements TypeConverter<CharSequence, T>, ConversionDependant {
    private ConversionRepository repository;

    @Override
    public void setConversionRepository(final ConversionRepository repository) {
      this.repository = repository;
    }

    protected Trans[] convertModels(CharSequence from) {
      int index = CharSequenceTools.skipTo(from, 0, '{') + 1;
      final int modelsCount = Integer.parseInt(CharSequenceTools.cut(from, index, ',').toString().trim());
      final Trans[] models = new Trans[modelsCount];
      index = CharSequenceTools.skipTo(from, index, ',');
      try {
        for (int i = 0; i < models.length; i++) {
          final CharSequence modelCS = CharSequenceTools.cutBetween(from, index, '{', '}');
          {
            int mindex = 0;
            String modelClassName = CharSequenceTools.cut(modelCS, mindex, ',').toString().trim();
            Class<? extends Trans> elementClass = (Class<? extends Trans>) Class.forName(modelClassName);
            models[i] = repository.convert(CharSequenceTools.cutBetween(modelCS, mindex, '{', '}'), elementClass);
          }
          index += modelCS.length();
        }
      } catch (ClassNotFoundException e) {
        throw new RuntimeException("Element class not found!", e);
      }
      return models;
    }
  }

  public static class From extends BaseFrom<TransJoin> {
    @Override
    public TransJoin convert(CharSequence from) {
      return new TransJoin(convertModels(from));
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
