package com.spbsu.ml.io;

import com.spbsu.commons.func.Computable;
import com.spbsu.commons.func.types.ConversionDependant;
import com.spbsu.commons.func.types.ConversionPack;
import com.spbsu.commons.func.types.ConversionRepository;
import com.spbsu.commons.func.types.TypeConverter;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.ml.Func;
import com.spbsu.ml.Trans;
import com.spbsu.ml.func.FuncJoin;
import com.spbsu.ml.func.TransJoin;
import com.spbsu.ml.models.multilabel.MultiLabelBinarizedModel;

/**
 * User: qdeee
 * Date: 03.04.15
 */
public class MultiLabelBinarizedModelConversionPack implements ConversionPack<MultiLabelBinarizedModel, CharSequence> {
  public static class To implements TypeConverter<MultiLabelBinarizedModel, CharSequence>, ConversionDependant {
    private ConversionRepository repository;

    @Override
    public CharSequence convert(final MultiLabelBinarizedModel from) {
      final TransJoin internModel = from.getInternModel();
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
      final TransJoin internModel = repository.convert(from, TransJoin.class);
      final Func[] dirs = ArrayTools.map(internModel.dirs, Func.class, new Computable<Trans, Func>() {
        @Override
        public Func compute(final Trans argument) {
          return (Func) argument;
        }
      });
      return new MultiLabelBinarizedModel(new FuncJoin(dirs));
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
