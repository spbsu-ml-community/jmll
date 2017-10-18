package com.expleague.ml.io;

import com.expleague.commons.func.types.ConversionDependant;
import com.expleague.commons.func.types.ConversionPack;
import com.expleague.commons.func.types.ConversionRepository;
import com.expleague.commons.func.types.TypeConverter;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.seq.CharSeqReader;
import com.expleague.commons.seq.CharSeqTools;
import com.expleague.commons.seq.ReaderChopper;
import com.expleague.commons.seq.regexp.Alphabet;
import com.expleague.ml.models.hmm.HiddenMarkovModel;

/**
 * User: solar
 * Date: 25.09.17
 */
public class HMMConversionPack implements ConversionPack<HiddenMarkovModel, CharSequence> {
  public static class To implements TypeConverter <HiddenMarkovModel, CharSequence>, ConversionDependant {
    private ConversionRepository repository;

    @Override
    public void setConversionRepository(final ConversionRepository repository) {
      this.repository = repository;
    }

    @Override
    public CharSequence convert(final HiddenMarkovModel hmm) {
      StringBuilder result = new StringBuilder();
      result.append(hmm.states()).append("\n")
          .append(repository.convert(hmm.alpha(), CharSequence.class)).append("\n")
          .append(repository.convert(hmm.betta(), CharSequence.class));
      return result;
    }
  }

  public static class From implements TypeConverter<CharSequence, HiddenMarkovModel> , ConversionDependant {
    private ConversionRepository repository;

    @Override
    public void setConversionRepository(final ConversionRepository repository) {
      this.repository = repository;
    }

    @SuppressWarnings("unchecked")
    @Override
    public HiddenMarkovModel convert(final CharSequence source) {
      final ReaderChopper chopper = new ReaderChopper(new CharSeqReader(source));
      final int states = CharSeqTools.parseInt(chopper.chopQuiet('\n'));
      final Alphabet alpha = repository.convert(chopper.chopQuiet('\n'), Alphabet.class);
      final Vec betta = repository.convert(chopper.restQuiet(), Vec.class);
      return new HiddenMarkovModel(alpha, states, betta);
    }
  }

  @Override
  public Class<? extends TypeConverter<HiddenMarkovModel, CharSequence>> to() {
    return To.class;
  }

  @Override
  public Class<? extends TypeConverter<CharSequence, HiddenMarkovModel>> from() {
    return From.class;
  }
}
