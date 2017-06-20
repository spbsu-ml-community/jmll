package com.spbsu.direct;

import com.spbsu.commons.func.Action;
import com.spbsu.commons.io.codec.seq.ListDictionary;
import com.spbsu.commons.seq.CharSeq;
import com.spbsu.commons.seq.CharSeqTools;
import com.spbsu.commons.seq.IntSeq;
import com.spbsu.direct.gen.NaiveModel;

import java.io.IOException;
import java.io.StringReader;

import static com.spbsu.direct.Utils.convertToSeq;
import static com.spbsu.direct.Utils.dropUnknown;
import static com.spbsu.direct.Utils.normalizeQuery;

public class NaiveProcessor implements Action<CharSequence> {
  private final static Integer DEBUG_COUNT = 100_000;
  private final static Integer MAX_COUNT = 20_000_000;

  private Integer index = 0;

  private final ListDictionary<CharSeq> dictionary;
  private final NaiveModel model;

  private String firstQuery;

  NaiveProcessor(final NaiveModel model, final ListDictionary<CharSeq> dictionary) {
    this.model = model;
    this.dictionary = dictionary;
    this.firstQuery = null;
  }

  @Override
  public void invoke(CharSequence line) {
    try {
      if (line.length() == 0 || index.equals(MAX_COUNT)) {
        return;
      }

      final CharSequence[] parts = new CharSequence[2];

      if (CharSeqTools.split(line, '\t', parts).length != 2) {
        throw new IllegalArgumentException("Incorrect format!");
      }

      final String user = parts[0].toString();
      final String query = normalizeQuery(parts[1].toString());

      if (query == null) {
        return;
      }

      if (firstQuery == null) {
        firstQuery = query;
      } else {
        final IntSeq prevQSeq = dropUnknown(dictionary.parse(convertToSeq(firstQuery), model.freqs, model.totalFreq));
        final IntSeq currQSeq = dropUnknown(dictionary.parse(convertToSeq(query), model.freqs, model.totalFreq));

        if (prevQSeq != null && currQSeq != null) {
          model.processSeq(prevQSeq);
          model.processSeq(currQSeq);

          model.processGeneration(prevQSeq, currQSeq);

          if (++index % DEBUG_COUNT == 0) {
            System.out.println(String.format("Processed: %d", index));
          }
        }

        firstQuery = null;
      }
    } catch (Exception e) {
      System.out.println("Ooops! " + e.toString());
    }
  }
}
