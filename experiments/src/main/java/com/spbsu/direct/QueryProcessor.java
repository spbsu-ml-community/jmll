package com.spbsu.direct;

import com.spbsu.commons.func.Action;
import com.spbsu.commons.io.codec.seq.ListDictionary;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.seq.CharSeq;
import com.spbsu.commons.seq.CharSeqTools;
import com.spbsu.commons.seq.IntSeq;
import com.spbsu.direct.gen.NaiveModel;
import com.spbsu.direct.gen.SimpleGenerativeModel;

import java.io.*;

import static com.spbsu.direct.Utils.convertToSeq;
import static com.spbsu.direct.Utils.dropUnknown;
import static com.spbsu.direct.Utils.normalizeQuery;

public class QueryProcessor implements Action<CharSequence> {
  private final static int DUMP_FREQ = 100_000;
  private final static double ALPHA = 0.001;

  private volatile static int index;

  private final ListDictionary<CharSeq> dictionary;
  private final SimpleGenerativeModel model;
  private final NaiveModel naiveModel;

  private final String inputFile;

  private final FastRandom rand = new FastRandom();

  private int succeeded;

  private int count;
  private String user;
  private String prevQuery;
  private String selectedPrevQuery;
  private String selectedCurrQuery;

  private CharSequence prevLine; // debug
  private CharSequence selectedPrevLine; // debug
  private CharSequence selectedCurrLine; // debug

  private Writer debugOutput;

  private double modelLogP = 0;
  private double naiveModelLogP = 0;

  public QueryProcessor(final String inputFile,
                        final ListDictionary<CharSeq> dictionary,
                        final SimpleGenerativeModel model,
                        final NaiveModel naiveModel) {
    this.inputFile = inputFile;

    this.dictionary = dictionary;
    this.model = model;
    this.naiveModel = naiveModel;

    try {
      this.debugOutput = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("test_queries.txt"), "utf-8"));
    } catch (IOException e) {
      System.err.println(e.toString());
    }
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

    // TODO: process timestamp
    // final long ts = CharSeqTools.parseLong(parts[1]);

    final String user = parts[0].toString();
    final String query = normalizeQuery(parts[1].toString());

    if (query == null) {
      return;
    }

    if (this.user == null || !this.user.equals(user)) {
      if (this.user != null) {

        final IntSeq currQSeq = dropUnknown(dictionary.parse(convertToSeq(selectedCurrQuery), model.freqs, model.totalFreq));

        if (selectedPrevQuery == null) {
          /*if (currQSeq != null) {
            double logP = model.maxLogP(IntSeq.EMPTY, currQSeq);
            double naiveLogP = naiveModel.maxLogP(IntSeq.EMPTY, currQSeq);

            if (logP != Double.NEGATIVE_INFINITY && naiveLogP != Double.NEGATIVE_INFINITY) {
              modelLogP += logP;
              naiveModelLogP += naiveLogP;
              ++succeeded;
            }
          }*/
        } else {
          final IntSeq prevQSeq = dropUnknown(dictionary.parse(convertToSeq(selectedPrevQuery), model.freqs, model.totalFreq));

          if (currQSeq != null && prevQSeq != null) {
            // TODO: debug output
            if (index % DUMP_FREQ == 0) {
              Utils.Timer.clearStatistics();
              Utils.Timer.start("new block", true);
            }

            if (index % 1_000 == 0) {
              Utils.Timer.start("new small block", true);
            }

            double logP = model.maxLogP(prevQSeq, currQSeq);
            double naiveLogP = naiveModel.maxLogP(prevQSeq, currQSeq);

            try {
              debugOutput.write(String.format("\n%s\n%s\n", selectedPrevLine.toString(), selectedCurrLine.toString()));
              debugOutput.write(String.format("%f\n%f\n", logP, naiveLogP));

            } catch (IOException e) {
              System.err.println(e.toString());
            }

            if (logP != Double.NEGATIVE_INFINITY && naiveLogP != Double.NEGATIVE_INFINITY) {
              modelLogP += logP;
              naiveModelLogP += naiveLogP;
              ++succeeded;

              // TODO: debug output
              if (++index % 1000 == 0) {
                System.out.println(String.format("processed %d (succeeded %d)", index, succeeded));
                Utils.Timer.stop("processing", true);
              }

              if (index % DUMP_FREQ == 0) {
                Utils.Timer.stop("total", true);
                Utils.Timer.showStatistics("total");
                dump(model);

                succeeded = 0;
                modelLogP = 0;
                naiveModelLogP = 0;
              }
            }
          }
        }
      }

      this.user = user;

      count = 0;
      prevQuery = null;
      selectedPrevQuery = null;
      selectedCurrQuery = null;
    }

    // TODO: skip similar prev query
    if (prevQuery == null || prevQuery.compareTo(query) != 0) {
      ++count;
      if (rand.nextInt(count) == 0) {
        selectedPrevQuery = prevQuery;
        selectedCurrQuery = query;

        selectedPrevLine = prevLine;
        selectedCurrLine = line;
      }

      prevQuery = query;

      prevLine = line;
    }

    // TODO: else { update timestamp }
  }

  public void dump(final SimpleGenerativeModel model) {
    try (final Writer out = new OutputStreamWriter(new FileOutputStream("results-" + (index / DUMP_FREQ) + ".txt"))) {
      out.write(String.format("Count:\t%d\nModel:\t%f\nNaive:\t%f\n\n", succeeded, Math.exp(-modelLogP / succeeded), Math.exp(-naiveModelLogP / succeeded)));
    } catch (IOException e) {
      e.printStackTrace();
    }
  }
}
