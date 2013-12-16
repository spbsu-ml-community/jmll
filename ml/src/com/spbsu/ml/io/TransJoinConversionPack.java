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
  public static class To implements TypeConverter<TransJoin, CharSequence>, ConversionDependant {
    private ConversionRepository repository;

    @Override
    public CharSequence convert(TransJoin from) {
      StringBuilder builder = new StringBuilder();
      builder.append("{").append(from.dirs.length).append(",\n");
      for (int i = 0; i < from.dirs.length; i++) {
        builder.append("{");
        Trans model = from.dirs[i];
        builder.append(from.dirs[i].getClass().getCanonicalName()).append(",\n");
        builder.append(repository.convert(model, CharSequence.class));
        builder.append("},\n");
      }
      builder.delete(builder.length() - 1, builder.length());
      builder.append("}");
      return builder;
    }

    @Override
    public void setConversionRepository(ConversionRepository repository) {
      this.repository = repository;
    }
  }

  public static class From implements TypeConverter<CharSequence, TransJoin>, ConversionDependant {
    private ConversionRepository repository;

    @Override
    public TransJoin convert(CharSequence from) {
      int index = 0;
      CharSequenceTools.cut(from, index, '{');
      Trans[] models = new Trans[Integer.parseInt(CharSequenceTools.cut(from, index, ',').toString().trim())];
      index = CharSequenceTools.skipTo(from, index, ',');
      try {
        for (int i = 0; i < models.length; i++) {
          index = CharSequenceTools.skipTo(from, index, '{');
          final CharSequence modelCS = CharSequenceTools.cutClose(from, index, '{', '}');
          {
            int mindex = 0;
            String modelClassName = CharSequenceTools.cut(modelCS, mindex, ',').toString().trim();
            Class<? extends Trans> elementClass = (Class<? extends Trans>) Class.forName(modelClassName);
            mindex = CharSequenceTools.skipTo(modelCS, mindex, ',');
            mindex = CharSequenceTools.skipTo(modelCS, mindex, '{');
            models[i] = repository.convert(CharSequenceTools.cutClose(modelCS, mindex, '{', '}'), elementClass);
          }
          index += modelCS.length();
        }
      } catch (ClassNotFoundException e) {
        throw new RuntimeException("Element class not found!", e);
      }
      return new TransJoin(models);
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
