package com.spbsu.direct;

import com.spbsu.commons.func.Action;
import com.spbsu.commons.io.StreamTools;
import com.spbsu.commons.io.codec.seq.ListDictionary;
import com.spbsu.commons.seq.CharSeq;
import com.spbsu.commons.seq.CharSeqTools;
import com.spbsu.direct.gen.NaiveModel;
import com.spbsu.direct.gen.SimpleGenerativeModel;
import gnu.trove.list.TIntList;
import gnu.trove.list.array.TIntArrayList;

import java.io.IOException;
import java.io.InputStreamReader;

import static com.spbsu.direct.Utils.convertToSeq;
import static com.spbsu.direct.Utils.loadDictionaryWithFreqs;
import static com.spbsu.direct.Utils.normalizeQuery;


public class Test {
  public static void main(String[] args) throws IOException {
    final TIntList freqs = new TIntArrayList();

    Utils.Timer.start("reading dictionary...", true);
    final ListDictionary<CharSeq> dict = loadDictionaryWithFreqs(args[0], freqs);
    Utils.Timer.stop("reading", true);

    final SimpleGenerativeModel model = new SimpleGenerativeModel(dict, freqs);
    Utils.Timer.start("loading model... ", true);
    //model.load(args[1]);
    Utils.Timer.stop("loading", true);

    final NaiveModel naiveModel = new NaiveModel(dict, freqs);
    Utils.Timer.start("naive model is learning...", true);
    CharSeqTools.processLines(StreamTools.openTextFile(args[2]), new NaiveProcessor(naiveModel, dict));
    Utils.Timer.stop("learning", true);

    CharSeqTools.processLines(new InputStreamReader(System.in, StreamTools.UTF), (Action<CharSequence>) arg -> {
      naiveModel.printTop(arg.toString(), 5);
    });

    /*
    try {
      CharSeqTools.processLines(StreamTools.openTextFile(args[3]), new QueryProcessor(args[3], dict, model, naiveModel));
    } catch (Exception e) {
      System.err.println(String.format("Failed to process %s: %s", args[3], e.toString()));
    }*/
  }
}
