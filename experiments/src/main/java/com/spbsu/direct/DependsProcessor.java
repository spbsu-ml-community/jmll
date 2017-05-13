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
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static com.spbsu.direct.Utils.convertToSeq;
import static com.spbsu.direct.Utils.dropUnknown;
import static com.spbsu.direct.Utils.normalizeQuery;


public class DependsProcessor implements Action<CharSequence> {
  private final static int DUMP_FREQ = 100_000;
  private final static double ALPHA = 0.5;
  private final static int MAX_ATTEMPTS_COUNT = 5;

  private volatile static int index;

  private final ListDictionary<CharSeq> dictionary;
  private final SimpleGenerativeModel model;
  private final String inputFile;

  private final Random rand = new Random();

  private String user;
  private final List<String> queries;

  public DependsProcessor(final String inputFile,
                          final ListDictionary<CharSeq> dictionary,
                          final SimpleGenerativeModel model) {
    this.inputFile = inputFile;

    this.dictionary = dictionary;
    this.model = model;

    this.queries = new ArrayList<>();
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
      if (!this.queries.isEmpty()) {

        // TODO: debug output
        if (index % DUMP_FREQ == 0) {
          Utils.Timer.clearStatistics();
          Utils.Timer.start("new block", true);
        }

        if (index % 1_000 == 0) {
          Utils.Timer.start("new small block", true);
        }

        for (int attempt = 0; attempt < MAX_ATTEMPTS_COUNT; ++attempt) {
          final int index = rand.nextInt(this.queries.size()) - 1;
          final IntSeq currQSeq = dropUnknown(dictionary.parse(convertToSeq(queries.get(index + 1)), model.freqs, model.totalFreq));

          if (currQSeq == null) {
            continue;
          }

          if (index == -1) {
            model.processSeq(currQSeq);
            model.processGeneration(IntSeq.EMPTY, currQSeq, ALPHA);
            break;
          } else {
            final IntSeq prevQSeq = dropUnknown(dictionary.parse(convertToSeq(queries.get(index)), model.freqs, model.totalFreq));

            if (prevQSeq == null) {
              continue;
            }

            model.processSeq(prevQSeq);
            model.processSeq(currQSeq);
            model.processGeneration(prevQSeq, currQSeq, ALPHA);
            break;
          }
        }

        // TODO: debug output
        if (++index % 1000 == 0) {
          System.out.println(String.format("processed %d", index));
          Utils.Timer.stop("processing", true);
        }

        if (index % DUMP_FREQ == 0) {
          Utils.Timer.stop("total", true);
          Utils.Timer.showStatistics("total");
          dump(model);
        }
      }

      this.queries.clear();
      this.user = user;
    }

    this.queries.add(query);
  }

  public static void dump(final SimpleGenerativeModel model) {
    try (final Writer out = new OutputStreamWriter(new FileOutputStream("output-" + (index / DUMP_FREQ) + ".txt"))) {
      model.printProviders(out, true);
    } catch (IOException e) {
      e.printStackTrace();
    }
  }
}
