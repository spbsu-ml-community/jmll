package com.expleague.ml.methods.seq;

import com.expleague.commons.io.codec.seq.DictExpansion;
import com.expleague.commons.io.codec.seq.Dictionary;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.seq.Seq;
import com.expleague.ml.TargetFunc;
import com.expleague.ml.data.set.DataSet;
import com.expleague.ml.methods.SeqOptimization;
import com.fasterxml.jackson.annotation.JsonIgnore;

import java.io.PrintStream;
import java.util.Collection;
import java.util.HashSet;
import java.util.function.Function;

public class DictExpansionOptimization<T extends Comparable<T>, Loss extends TargetFunc>
    implements SeqOptimization<T, Loss> {

  private final SeqOptimization<Integer, Loss> optimization;
  private final int maxAlphabetSize;
  private Collection<T> alphabet;
  private PrintStream tracePrint;

  public DictExpansionOptimization(
      SeqOptimization<Integer, Loss> optimization,
      int maxAlphabetSize,
      Collection<T> alphabet,
      PrintStream tracePrint
  ) {
    this.optimization = optimization;
    this.maxAlphabetSize = maxAlphabetSize;
    this.alphabet = alphabet;
    this.tracePrint = tracePrint;
  }

  @Override
  public Function<Seq<T>, Vec> fit(DataSet<Seq<T>> learn, Loss loss) {
    final long startTime = System.nanoTime();

    Collection<T> realAlphabet = alphabet;
    if (alphabet == null) {
      realAlphabet = new HashSet<>();
      for (Seq<T> seq: learn) {
        for (T t: seq) {
          realAlphabet.add(t);
        }
      }
    }

    final DictExpansion<T> de = new DictExpansion<>(realAlphabet, maxAlphabetSize, tracePrint);

    for (int iter = 0; iter < 1000; iter++) {
      for (Seq<T> seq: learn) {
        de.accept(seq);
        if (de.result() != null && de.result().size() == maxAlphabetSize) {
          break;
        }
      }
    }

    final Dictionary<T> dict = de.result();

    if (dict.size() < maxAlphabetSize) {
      throw new IllegalStateException("Cannot build alphabet of size " + maxAlphabetSize + ". Actual size is " + dict.size());
    }

    if (tracePrint != null) {
      tracePrint.println("Time to build dictexpansion: " + (System.nanoTime() - startTime) / 1e9 + "s");
    }

    final DataSet<Seq<Integer>> expandedLearn = new DataSet.Stub<Seq<Integer>>(null) {

      @Override
      public Seq<Integer> at(int i) {
        return dict.parse(learn.at(i));
      }

      @Override
      public int length() {
        return learn.length();
      }

      @Override
      public Class<? extends Seq<Integer>> elementType() {
        return null;
      }
    };

    Function<Seq<Integer>, Vec> model = optimization.fit(expandedLearn, loss);
    return new DictExpansionModel<>(model, dict);
  }

  static class DictExpansionModel<T extends Comparable<T>> implements Function<Seq<T>, Vec> {
    private Function<Seq<Integer>, Vec> model;
    @JsonIgnore
    private Dictionary<T> dict;

    public DictExpansionModel(Function<Seq<Integer>, Vec> model, Dictionary<T> dict) {
      this.model = model;
      this.dict = dict;
    }

    public Function<Seq<Integer>, Vec> getModel() {
      return model;
    }

    public void setModel(Function<Seq<Integer>, Vec> model) {
      this.model = model;
    }

    public Dictionary<T> getDict() {
      return dict;
    }

    public void setDict(Dictionary<T> dict) {
      this.dict = dict;
    }

    @Override
    public Vec apply(Seq<T> seq) {
      return model.apply(dict.parse(seq));
    }
  }
}
