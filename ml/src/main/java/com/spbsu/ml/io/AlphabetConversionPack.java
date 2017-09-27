package com.spbsu.ml.io;

import com.spbsu.commons.func.types.ConversionDependant;
import com.spbsu.commons.func.types.ConversionPack;
import com.spbsu.commons.func.types.ConversionRepository;
import com.spbsu.commons.func.types.TypeConverter;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.seq.CharSeq;
import com.spbsu.commons.seq.CharSeqReader;
import com.spbsu.commons.seq.CharSeqTools;
import com.spbsu.commons.seq.ReaderChopper;
import com.spbsu.commons.seq.regexp.Alphabet;
import com.spbsu.commons.seq.regexp.DynamicCharAlphabet;
import com.spbsu.ml.models.hmm.HiddenMarkovModel;

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
