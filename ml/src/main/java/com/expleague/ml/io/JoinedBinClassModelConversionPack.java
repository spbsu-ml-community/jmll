package com.expleague.ml.io;

import com.expleague.commons.func.types.ConversionPack;
import com.expleague.commons.func.types.ConversionDependant;
import com.expleague.commons.func.types.ConversionRepository;
import com.expleague.commons.func.types.TypeConverter;
import com.expleague.ml.func.FuncJoin;
import com.expleague.ml.models.multiclass.JoinedBinClassModel;

/**
 * User: qdeee
 * Date: 03.04.15
 */
public class JoinedBinClassModelConversionPack implements ConversionPack<JoinedBinClassModel, CharSequence> {
  public static class To implements TypeConverter<JoinedBinClassModel, CharSequence>, ConversionDependant {
    private ConversionRepository repository;

    @Override
    public CharSequence convert(final JoinedBinClassModel from) {
      final FuncJoin internModel = from.getInternModel();
      return repository.convert(internModel, CharSequence.class);
    }

    @Override
    public void setConversionRepository(final ConversionRepository repository) {
      this.repository = repository;
    }

  }
  public static class From implements TypeConverter<CharSequence, JoinedBinClassModel>, ConversionDependant {
    private ConversionRepository repository;

    @Override
    public JoinedBinClassModel convert(final CharSequence from) {
      final FuncJoin internModel = repository.convert(from, FuncJoin.class);
      return new JoinedBinClassModel(internModel);
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
