package com.expleague.ml.io;

import com.expleague.commons.func.types.ConversionPack;
import com.expleague.commons.func.types.ConversionRepository;
import com.expleague.commons.func.types.TypeConverter;
import com.expleague.ml.func.FuncJoin;
import com.expleague.commons.func.types.ConversionDependant;
import com.expleague.ml.models.multilabel.MultiLabelBinarizedModel;

/**
 * User: qdeee
 * Date: 03.04.15
 */
public class MultiLabelBinarizedModelConversionPack implements ConversionPack<MultiLabelBinarizedModel, CharSequence> {
  public static class To implements TypeConverter<MultiLabelBinarizedModel, CharSequence>, ConversionDependant {
    private ConversionRepository repository;

    @Override
    public CharSequence convert(final MultiLabelBinarizedModel from) {
      final FuncJoin internModel = from.getInternModel();
      return repository.convert(internModel, CharSequence.class);
    }

    @Override
    public void setConversionRepository(final ConversionRepository repository) {
      this.repository = repository;
    }

  }
  public static class From implements TypeConverter<CharSequence, MultiLabelBinarizedModel>, ConversionDependant {
    private ConversionRepository repository;

    @Override
    public MultiLabelBinarizedModel convert(final CharSequence from) {
      final FuncJoin internModel = repository.convert(from, FuncJoin.class);
      return new MultiLabelBinarizedModel(internModel);
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
