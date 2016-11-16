import automaton.DFA;
import com.spbsu.commons.seq.IntSeq;
import com.spbsu.commons.seq.Seq;
import com.spbsu.commons.seq.regexp.Alphabet;
import com.spbsu.commons.seq.regexp.Matcher;
import gnu.trove.list.TIntList;
import gnu.trove.list.array.TIntArrayList;
import learning.IncrementalAutomatonBuilder;
import learning.NonOptimizedCostFunction;
import learning.OptimizedCostFunction;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.function.Function;

import static org.junit.Assert.assertEquals;

public class TestIncrementalBuilder {
  private  Random r = new Random(239);

  @Test
  public void testMod3() {
    final List<Seq<Integer>> data = new ArrayList<>();
    final TIntList classes = new TIntArrayList();
    Function<TIntList, Integer> classifier = word -> word.sum() % 3;

    generateData(data, classes, classifier, 100000, 10, 50, 3, 2);
    DFA<Integer> automaton = new IncrementalAutomatonBuilder<Integer>(new OptimizedCostFunction<>())
            .buildAutomaton(data, classes, new IntAlphabet(2), 3, 30);
    data.clear();
    classes.clear();
    generateData(data, classes, classifier, 500000, 10, 50, 3, 2);

    System.out.println("testMod3: precision " + getAccuracy(data, classes, automaton));
    System.out.println(automaton.toString());
  }

  @Test
  public void testNoSubsequentThreeZeroes() {
    final List<Seq<Integer>> data = new ArrayList<>();
    final TIntList classes = new TIntArrayList();

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

    generateData(data, classes, classifier, 100000, 10, 50, 2, 2);
    DFA<Integer> automaton = new IncrementalAutomatonBuilder<Integer>(new OptimizedCostFunction<>())
            .buildAutomaton(data, classes, new IntAlphabet(2), 2, 30);
    data.clear();
    classes.clear();
    generateData(data, classes, classifier, 500000, 10, 50, 2, 2);
    System.out.println("testNoSubsequentThreeZeroes: precision " + getAccuracy(data, classes, automaton));
    System.out.println(automaton.toString());

  }


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
    DFA<Integer> automaton = new IncrementalAutomatonBuilder<Integer>(new OptimizedCostFunction<>())
            .buildAutomaton(data, classes, new IntAlphabet(2), 2, 30);
    data.clear();
    classes.clear();
    generateData(data, classes, classifier, 500000, 10, 50, 2, 2);
    System.out.println("testEndsWith: precision " + getAccuracy(data, classes, automaton));
    System.out.println(automaton.toString());

  }

  private double getAccuracy(List<Seq<Integer>> data, TIntList classes, DFA<Integer> automaton) {
    int matchCnt = 0;
    for (int i = 0; i < data.size(); i++) {
      int clazz = automaton.getWordClass(data.get(i));
      if (clazz == classes.get(i)) {
        matchCnt++;
      }
    }
    return 1.0 * matchCnt / data.size();
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
    DFA<Integer> automaton = new IncrementalAutomatonBuilder<Integer>(new OptimizedCostFunction<>())
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
    DFA<Integer> optimized = new IncrementalAutomatonBuilder<Integer>(new OptimizedCostFunction<>())
            .buildAutomaton(data, classes, alphabet, 3, 20);
    DFA<Integer> nonOptimized = new IncrementalAutomatonBuilder<Integer>(new NonOptimizedCostFunction<>())
            .buildAutomaton(data, classes, alphabet, 3, 20);
    assertEquals(optimized, nonOptimized);
  }

  private void generateData(List<Seq<Integer>> data, TIntList classes, Function<TIntList, Integer> classifier,
                            int sampleCount, int minLen, int maxLen, int classCount, int alphabetSize) {
    int[] classSamplesCount = new int[classCount];
    while (data.size() < sampleCount) {
      int len = r.nextInt(maxLen - minLen + 1) + minLen;
      TIntList sample = new TIntArrayList();
      for (int i = 0; i < len; i++) {
        sample.add(r.nextInt(alphabetSize));
      }
      int clazz = classifier.apply(sample);
      if (classSamplesCount[clazz] <= data.size() / classCount) {
        sample.add(alphabetSize);
        data.add(new IntSeq(sample.toArray()));
        classes.add(clazz);
        classSamplesCount[clazz]++;
      }
    }
  }

  private class IntAlphabet implements Alphabet<Integer> {
    private final int size;

    public IntAlphabet(int size) {
      this.size = size;
    }

    @Override
    public int size() {
      return size;
    }

    @Override
    public int getOrder(Matcher.Condition<Integer> c) {
      if (c == Matcher.Condition.ANY) {
        return size;
      }
      
      if (!(c instanceof IntCondition)) {
        throw new IllegalArgumentException("Not a int condition");
      }
      return ((IntCondition) c).my;
    }

    @Override
    public Matcher.Condition<Integer> get(int i) {
      if (i == size) {
        //noinspection unchecked
        return Matcher.Condition.ANY;
      }
      return new IntCondition(i);
    }

    @Override
    public Matcher.Condition<Integer> getByT(Integer i) {
      return new IntCondition(i);
    }

    @Override
    public Integer getT(Matcher.Condition condition) {
      if (condition == Matcher.Condition.ANY) {
        return size;
      } else if (condition instanceof IntCondition) {
        return ((IntCondition) condition).my;
      }
      return null;
    }

    @Override
    public int index(Integer integer) {
      return integer;
    }

    @Override
    public int index(Seq<Integer> seq, int index) {
      return seq.at(index);
    }

    private class IntCondition implements Matcher.Condition<Integer> {
      private final int my;

      IntCondition(int my) {
        this.my = my;
      }
      
      @Override
      public boolean is(Integer frag) {
        return frag == my;
      }

      @Override
      public String toString() {
        return Integer.toString(my);
      }

      @Override
      public boolean equals(Object o) {
        if (o == this) {
          return true;
        }
        if (o == null || !(o instanceof IntCondition)) {
          return false;
        }

        final IntCondition c = (IntCondition) o;
        return c.my == my;
      }

      @Override
      public int hashCode() {
        return Integer.hashCode(my);
      }
    }
  }

}
