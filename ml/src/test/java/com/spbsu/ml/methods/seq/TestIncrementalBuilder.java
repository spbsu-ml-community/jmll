package com.spbsu.ml.methods.seq;

import com.spbsu.commons.func.Computable;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.seq.IntSeq;
import com.spbsu.commons.seq.Seq;
import com.spbsu.commons.util.Pair;
import com.spbsu.ml.data.set.DataSet;
import gnu.trove.list.TIntList;
import gnu.trove.list.array.TIntArrayList;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.function.Function;

import static org.junit.Assert.assertEquals;

public class TestIncrementalBuilder {
  private  Random r = new Random(239);
/*
  @Test
  public void testMod3() {

    Function<TIntList, Integer> classifier = word -> word.sum() % 3;

    Pair<DataSet<Seq<Integer>>, Vec> data = generateData(classifier, 100000, 10, 50, 3, 2);
    Computable<Seq<Integer>, Vec> automaton = new IncrementalAutomatonBuilder<>(new IntAlphabet(2), new OptimizedStateEvaluation<>())
            .fit(data.first, new L2(data.second, null));
    data = generateData(classifier, 500000, 10, 50, 3, 2);

    System.out.println("testMod3: precision " + getAccuracy(data.first, data.second, computable));
    System.out.println(automaton.toString());
  }*/

  @Test
  public void testNoSubsequentThreeZeroes() {
    Function<TIntList, Integer> classifier = word -> {
      for (int i = 0; i < word.size() - 2; i++) {
        boolean matches = true;
        for (int j = 0; j < 3; j++) {
          if (word.get(i + j) != 0) {
            matches = false;
            break;
          }
        }
        if (matches) {
          return 0;
        }
      }
      return 1;
    };

    Pair<DataSet<Seq<Integer>>, Vec> data = generateData(classifier, 100000, 10, 50, 2, 2);
//    Computable<Seq<Integer>, Vec> computable = new IncrementalAutomatonBuilder<>(new IntAlphabet(2), new OptimizedStateEvaluation<>())
  //          .fit(data.first, new L2(data.second, null));
  //  data = generateData(classifier, 500000, 10, 50, 2, 2);
    //System.out.println("testNoSubsequentThreeZeroes: precision " + getAccuracy(data.first, data.second, computable));
    //System.out.println(automaton.toString());

  }
/*

  @Test
  public void testEndsWith() {
    final List<Seq<Integer>> data = new ArrayList<>();
    final TIntList classes = new TIntArrayList();
    final int[] end = {1, 0, 0, 0, 1};

    Function<TIntList, Integer> classifier = word -> {
      boolean matches = true;
      for (int i = 0; i < end.length; i++) {
        if (word.get(word.size() - end.length + i) != end[i]) {
          matches = false;
          break;
        }
      }
      return matches ? 1 : 0;
    };

    generateData(data, classes, classifier, 100000, 10, 50, 2, 2);
    DFA<Integer> automaton = new IncrementalAutomatonBuilder<Integer>(new IntAlphabet(2), new OptimizedStateEvaluation<>())
            .buildAutomaton(data, classes, , 2, 30);
    data.clear();
    classes.clear();
    generateData(data, classes, classifier, 500000, 10, 50, 2, 2);
    System.out.println("testEndsWith: precision " + getAccuracy(data, classes, automaton));
    System.out.println(automaton.toString());

  }

  @Test
  public void testFindSubstring() {
    final List<Seq<Integer>> data = new ArrayList<>();
    final TIntList classes = new TIntArrayList();
    final int[] substring = {0, 0, 1, 0, 1};

    Function<TIntList, Integer> classifier = word -> {
      for (int i = 0; i < word.size() - substring.length + 1; i++) {
        boolean matches = true;
        for (int j = i; j < i + substring.length; j++) {
          if (word.get(j) != substring[j - i]) {
            matches = false;
            break;
          }
        }
        if (matches) {
          return 1;
        }
      }
      return 0;
    };

    generateData(data, classes, classifier, 100000, 10, 50, 2, 2);
    DFA<Integer> automaton = new IncrementalAutomatonBuilder<Integer>(new OptimizedStateEvaluation<>())
            .buildAutomaton(data, classes, new IntAlphabet(2), 2, 30);
    data.clear();
    classes.clear();
    generateData(data, classes, classifier, 500000, 10, 50, 2, 2);

    System.out.println("testFindSubstring: precision " + getAccuracy(data, classes, automaton));
    System.out.println(automaton.toString());
  }

  @Test
  public void testCostFunction() {
    final List<Seq<Integer>> data = new ArrayList<>();
    final TIntList classes = new TIntArrayList();
    final Alphabet<Integer> alphabet = new IntAlphabet(3);
    Function<TIntList, Integer> classifier = (word) -> word.sum() % 3 ;
    generateData(data, classes, classifier, 5000, 10, 50, 3, 3);
    DFA<Integer> optimized = new IncrementalAutomatonBuilder<Integer>(new OptimizedStateEvaluation<>())
            .buildAutomaton(data, classes, alphabet, 3, 20);
    DFA<Integer> nonOptimized = new IncrementalAutomatonBuilder<Integer>(new NonOptimizedStateEvaluation<>())
            .buildAutomaton(data, classes, alphabet, 3, 20);
    assertEquals(optimized, nonOptimized);
  }
*/
private double getAccuracy(DataSet<Seq<Integer>> data, Vec target, Computable<Seq<Integer>, Vec> computable) {
  int matchCnt = 0;
  for (int i = 0; i < data.length(); i++) {
    if (Math.round(computable.compute(data.at(i)).at(0)) == target.at(i)) {
      matchCnt++;
    }
  }
  return 1.0 * matchCnt / data.length();
}

  private Pair<DataSet<Seq<Integer>>, Vec> generateData(Function<TIntList, Integer> classifier,
                                                        int sampleCount, int minLen, int maxLen, int classCount, int alphabetSize) {
    int[] classSamplesCount = new int[classCount];
    final List<Seq<Integer>> data = new ArrayList<>(sampleCount);
    int[] target = new int[sampleCount];
    while (data.size() < sampleCount) {
      int len = r.nextInt(maxLen - minLen + 1) + minLen;
      TIntList sample = new TIntArrayList();
      for (int i = 0; i < len; i++) {
        sample.add(r.nextInt(alphabetSize));
      }
      int clazz = classifier.apply(sample);
      if (classSamplesCount[clazz] <= data.size() / classCount) {
        target[data.size()] = clazz;
        data.add(new IntSeq(sample.toArray()));
        classSamplesCount[clazz]++;
      }
    }
    DataSet<Seq<Integer>> dataSet = new DataSet.Stub<Seq<Integer>>(null) {
      @Override
      public Seq<Integer> at(int i) {
        return data.get(i);
      }

      @Override
      public int length() {
        return data.size();
      }

      @Override
      public Class elementType() {
        return Seq.class;
      }
    };
    return new Pair<>(dataSet, VecTools.fromIntSeq(new IntSeq(target)));
  }
}
