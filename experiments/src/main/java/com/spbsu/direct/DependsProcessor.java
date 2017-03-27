package com.spbsu.direct;

import com.spbsu.commons.func.Action;
import com.spbsu.commons.io.codec.seq.ListDictionary;
import com.spbsu.commons.seq.CharSeq;
import com.spbsu.commons.seq.CharSeqTools;
import com.spbsu.commons.seq.IntSeq;
import com.spbsu.direct.gen.SimpleGenerativeModel;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.util.concurrent.TimeUnit;

import static com.spbsu.direct.Utils.convertToSeq;
import static com.spbsu.direct.Utils.dropUnknown;
import static com.spbsu.direct.Utils.normalizeQuery;


public class DependsProcessor implements Action<CharSequence> {
  private final static int DUMP_FREQ = 10_000_000;
  private final static double ALPHA = 0.5;

  private volatile static int index;

  private final ListDictionary<CharSeq> dictionary;
  private final SimpleGenerativeModel model;
  private final String inputFile;

  private long ts;
  private String query;
  private String user;
  private IntSeq prevQSeq;

  public DependsProcessor(final String inputFile,
                          final ListDictionary<CharSeq> dictionary,
                          final SimpleGenerativeModel model) {
    this.inputFile = inputFile;

    this.dictionary = dictionary;
    this.model = model;
  }

  @Override
  public void invoke(CharSequence line) {
    final CharSequence[] parts = new CharSequence[3];

    // TODO: is it possible to replace with String::split?
    if (CharSeqTools.split(line, '\t', parts).length != 3) {
      throw new IllegalArgumentException("Each input line must contain <uid>\\t<ts>\\t<query> triplet. This one: [" + line + "]@" + inputFile + ":" + index + " does not.");
    }

    if (CharSeqTools.startsWith(parts[0], "uu/") || CharSeqTools.startsWith(parts[0], "r")) {
      return;
    }

    final long ts = CharSeqTools.parseLong(parts[1]);
    final String query = normalizeQuery(parts[2].toString());

    // TODO: use case?
    if (query == null || query.equals(this.query)) {
      this.ts = ts;
      return;
    }

    final IntSeq currentQSeq = dropUnknown(dictionary.parse(convertToSeq(query), model.freqs, model.totalFreq));

    if (currentQSeq == null) {
      this.ts = ts;
      return;
    }

    model.processSeq(currentQSeq);
    final String prev = parts[0].equals(this.user) && ts - this.ts < TimeUnit.MINUTES.toSeconds(30) ? this.query : null;
    this.query = parts[2].toString();
    this.user = parts[0].toString();
    this.ts = ts;

    if (prev != null && prevQSeq != null) {
      model.processGeneration(prevQSeq, currentQSeq, ALPHA);
    }

    prevQSeq = currentQSeq;

    if (++index % DUMP_FREQ == 0) {
      dump(model);
    }
  }

  public static void dump(final SimpleGenerativeModel model) {
    try (final Writer out = new OutputStreamWriter(new FileOutputStream("output-" + (index / DUMP_FREQ) + ".txt"))) {
      model.print(out, true);
    } catch (IOException e) {
      e.printStackTrace();
    }
  }
}
