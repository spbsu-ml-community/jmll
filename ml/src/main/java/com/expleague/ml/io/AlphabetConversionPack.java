package com.expleague.ml.io;

import com.expleague.commons.func.types.ConversionPack;
import com.expleague.commons.func.types.ConversionDependant;
import com.expleague.commons.func.types.ConversionRepository;
import com.expleague.commons.func.types.TypeConverter;
import com.expleague.commons.seq.CharSeqReader;
import com.expleague.commons.seq.CharSeqTools;
import com.expleague.commons.seq.ReaderChopper;
import com.expleague.commons.seq.regexp.Alphabet;
import com.expleague.commons.seq.regexp.DynamicCharAlphabet;

/**
 * User: solar
 * Date: 25.09.17
 */
public class AlphabetConversionPack implements ConversionPack<Alphabet, CharSequence> {
  public static class To implements TypeConverter <Alphabet, CharSequence>, ConversionDependant {
    private ConversionRepository repository;

    @Override
    public void setConversionRepository(final ConversionRepository repository) {
      this.repository = repository;
    }

    @Override
    public CharSequence convert(final Alphabet alpha) {
      if (alpha instanceof DynamicCharAlphabet) {
        StringBuilder result = new StringBuilder();
        final DynamicCharAlphabet charAlphabet = (DynamicCharAlphabet) alpha;

        result.append(alpha.getClass().getName()).append(":")
            .append(charAlphabet.chars());
        return result;

      }
      throw new UnsupportedOperationException();
    }
  }

  public static class From implements TypeConverter<CharSequence, Alphabet> , ConversionDependant {
    private ConversionRepository repository;

    @Override
    public void setConversionRepository(final ConversionRepository repository) {
      this.repository = repository;
    }

    @SuppressWarnings("unchecked")
    @Override
    public Alphabet convert(final CharSequence source) {
      final ReaderChopper chopper = new ReaderChopper(new CharSeqReader(source));
      if (CharSeqTools.equals(DynamicCharAlphabet.class.getName(), chopper.chopQuiet(':'))) {
        return new DynamicCharAlphabet(chopper.restQuiet().toCharArray());
      }
      throw new UnsupportedOperationException();
    }
  }

  @Override
  public Class<? extends TypeConverter<Alphabet, CharSequence>> to() {
    return To.class;
  }

  @Override
  public Class<? extends TypeConverter<CharSequence, Alphabet>> from() {
    return From.class;
  }
}
