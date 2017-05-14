package com.spbsu.direct;

import com.spbsu.commons.func.Action;
import com.spbsu.commons.io.codec.seq.ListDictionary;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.seq.CharSeq;
import com.spbsu.commons.seq.CharSeqTools;
import com.spbsu.commons.seq.IntSeq;
import com.spbsu.direct.gen.SimpleGenerativeModel;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static com.spbsu.direct.Utils.convertToSeq;
import static com.spbsu.direct.Utils.dropUnknown;
import static com.spbsu.direct.Utils.normalizeQuery;


public class DependsProcessor implements Action<CharSequence> {
  private final static int DUMP_FREQ = 100_000;
  private final static double ALPHA = 0.5;

  private volatile static int index;

  private final ListDictionary<CharSeq> dictionary;
  private final SimpleGenerativeModel model;
  private final String inputFile;

  private final FastRandom rand = new FastRandom();

  private int succeeded;

  private int count;
  private String user;
  private String prevQuery;
  private String selectedPrevQuery;
  private String selectedCurrQuery;

  public DependsProcessor(final String inputFile,
                          final ListDictionary<CharSeq> dictionary,
                          final SimpleGenerativeModel model) {
    this.inputFile = inputFile;

    this.dictionary = dictionary;
    this.model = model;
  }

  // TODO: process timestamps
  // TODO: process the last user
  @Override
  public void invoke(CharSequence line) {
    final CharSequence[] parts = new CharSequence[2];

    if (CharSeqTools.split(line, '\t', parts).length != 2) {
      throw new IllegalArgumentException("Each input line must contain <uid>\\t<ts>\\t<query> triplet. This one: [" + line + "]@" + inputFile + ":" + index + " does not.");
    }

    if (CharSeqTools.startsWith(parts[0], "uu/") || CharSeqTools.startsWith(parts[0], "r")) {
      return;
    }

    //final long ts = CharSeqTools.parseLong(parts[1]);

    final String user = parts[0].toString();
    final String query = normalizeQuery(parts[1].toString());

    if (this.user == null || !this.user.equals(user)) {
      if (this.user != null) {
        // TODO: debug output
        if (index % DUMP_FREQ == 0) {
          Utils.Timer.clearStatistics();
          Utils.Timer.start("new block", true);
        }

        if (index % 1_000 == 0) {
          Utils.Timer.start("new small block", true);
        }

        final IntSeq currQSeq = dropUnknown(dictionary.parse(convertToSeq(selectedCurrQuery), model.freqs, model.totalFreq));

        if (selectedPrevQuery == null) {
          if (currQSeq != null) {
            model.processSeq(currQSeq);
            model.processGeneration(IntSeq.EMPTY, currQSeq, ALPHA);

            ++succeeded;
          }
        } else {
          final IntSeq prevQSeq = dropUnknown(dictionary.parse(convertToSeq(selectedPrevQuery), model.freqs, model.totalFreq));

          if (currQSeq != null && prevQSeq != null) {
            model.processSeq(prevQSeq);
            model.processSeq(currQSeq);
            model.processGeneration(prevQSeq, currQSeq, ALPHA);

            ++succeeded;
          }
        }

        // TODO: debug output
        if (++index % 1000 == 0) {
          System.out.println(String.format("processed %d (succeeded %d)", index, succeeded));
          Utils.Timer.stop("processing", true);
        }

        if (index % DUMP_FREQ == 0) {
          Utils.Timer.stop("total", true);
          Utils.Timer.showStatistics("total");
          dump(model);
        }
      }

      this.user = user;

      count = 0;
      prevQuery = null;
      selectedPrevQuery = null;
      selectedCurrQuery = null;
    }

    ++count;
    if (rand.nextInt(count) == 0) {
      selectedPrevQuery = prevQuery;
      selectedCurrQuery = query;
    }

    prevQuery = query;
  }

  public static void dump(final SimpleGenerativeModel model) {
    try (final Writer out = new OutputStreamWriter(new FileOutputStream("output-" + (index / DUMP_FREQ) + ".txt"))) {
      model.printProviders(out, true);
    } catch (IOException e) {
      e.printStackTrace();
    }
  }
}
