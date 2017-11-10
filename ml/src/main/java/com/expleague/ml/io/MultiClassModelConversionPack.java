package com.expleague.ml.io;

import com.expleague.commons.func.types.ConversionPack;
import com.expleague.commons.func.types.ConversionRepository;
import com.expleague.commons.math.Func;
import com.expleague.commons.math.Trans;
import com.expleague.ml.models.MultiClassModel;
import com.expleague.commons.func.types.ConversionDependant;
import com.expleague.commons.func.types.TypeConverter;
import com.expleague.commons.util.ArrayTools;
import com.expleague.ml.func.TransJoin;

/**
 * User: starlight
 * Date: 17.04.14
 */
public class MultiClassModelConversionPack implements ConversionPack<MultiClassModel, CharSequence> {
  public static class To implements TypeConverter<MultiClassModel, CharSequence>, ConversionDependant {
    private ConversionRepository repository;

    @Override
    public CharSequence convert(final MultiClassModel from) {
      final Trans internModel = from.getInternModel();
      return repository.convert(internModel, CharSequence.class);
    }

    @Override
    public void setConversionRepository(final ConversionRepository repository) {
      this.repository = repository;
    }
  }

  public static class From implements TypeConverter<CharSequence, MultiClassModel>, ConversionDependant {
    private ConversionRepository repository;

    @Override
    public MultiClassModel convert(final CharSequence from) {
      final TransJoin internModel = repository.convert(from, TransJoin.class);
      final Func[] dirs = ArrayTools.map(internModel.dirs, Func.class, argument -> (Func) argument);
      return new MultiClassModel(dirs);
    }

    @Override
    public void setConversionRepository(final ConversionRepository repository) {
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
