package com.spbsu.ml.io;

import com.spbsu.commons.func.types.ConversionDependant;
import com.spbsu.commons.func.types.ConversionPack;
import com.spbsu.commons.func.types.ConversionRepository;
import com.spbsu.commons.func.types.TypeConverter;
import com.spbsu.ml.VecFunc;
import com.spbsu.commons.text.CharSequenceTools;
import com.spbsu.ml.Func;
import com.spbsu.ml.models.AdditiveMultiClassModel;

import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.text.MessageFormat;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * User: solar
 * Date: 12.08.13
 * Time: 17:16
 */
public class AdditiveMultiClassModelConversionPack implements ConversionPack<AdditiveMultiClassModel, CharSequence> {
  private static final MessageFormat HEADER = new MessageFormat("size: {0}, step: {1}, element: {2}", Locale.US);
  static {
    DecimalFormat format = new DecimalFormat();
    format.setDecimalSeparatorAlwaysShown(false);
    format.setGroupingUsed(false);
    format.setDecimalFormatSymbols(new DecimalFormatSymbols(Locale.US));
    HEADER.setFormat(0, format);
    HEADER.setFormat(1, format);
  }

  public static class To implements TypeConverter<AdditiveMultiClassModel, CharSequence>, ConversionDependant {
    private ConversionRepository repository;

    @Override
    public CharSequence convert(AdditiveMultiClassModel from) {
      StringBuilder builder = new StringBuilder();
      builder.append(HEADER.format(new Object[]{from.models.size(), from.step, from.models.size() > 0 ? from.models.get(0).getClass().getCanonicalName() : "none"}));
      builder.append("\n");
      builder.append("\n");
      for (int i = 0; i < from.models.size(); i++) {
        Func model = (Func) from.models.get(i);
        builder.append(repository.convert(model, CharSequence.class));
        builder.append("\n");
      }
      builder.delete(builder.length() - 1, builder.length());
      return builder;
    }

    @Override
    public void setConversionRepository(ConversionRepository repository) {
      this.repository = repository;
    }
  }

  public static class From implements TypeConverter<CharSequence, AdditiveMultiClassModel>, ConversionDependant {
    private ConversionRepository repository;

    @Override
    public AdditiveMultiClassModel convert(CharSequence from) {
      CharSequence[] elements = CharSequenceTools.split(from, "\n\n");
      List<VecFunc> models = new ArrayList<VecFunc>();
      double step;

      try {
        Object[] parse = HEADER.parse(elements[0].toString());
        Class<? extends VecFunc> elementClass = (Class<? extends VecFunc>) Class.forName(parse[2].toString());
        step = Double.parseDouble(parse[1].toString());
        int count = Integer.parseInt(parse[0].toString());
        for (int i = 0; i < count; i++) {
          models.add(repository.convert(elements[i + 1], elementClass));
        }
      } catch (ParseException e) {
        throw new RuntimeException("Invalid header!");
      } catch (ClassNotFoundException e) {
        throw new RuntimeException("Element class not found!", e);
      }
      return new AdditiveMultiClassModel(models, step);
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
