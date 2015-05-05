package com.spbsu.ml.io;

import com.spbsu.commons.func.types.ConversionDependant;
import com.spbsu.commons.func.types.ConversionPack;
import com.spbsu.commons.func.types.ConversionRepository;
import com.spbsu.commons.func.types.TypeConverter;
import com.spbsu.ml.func.FuncJoin;
import com.spbsu.ml.models.multiclass.JoinedBinClassModel;

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
