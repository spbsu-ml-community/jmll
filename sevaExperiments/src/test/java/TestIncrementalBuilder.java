import automaton.DFA;
import gnu.trove.list.TIntList;
import gnu.trove.list.array.TIntArrayList;
import learning.IncrementalAutomatonBuilder;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.function.Predicate;

public class TestIncrementalBuilder {
  private  Random r = new Random(239);

  @Test
  public void testMod3() {
    final List<TIntList> data = new ArrayList<>();
    final List<Boolean> classes = new ArrayList<>();
    Predicate<TIntList> predicate = word -> word.sum() % 3 == 0;

    generateData(data, classes, predicate, 100000, 10, 50, 2);
    DFA automaton = new IncrementalAutomatonBuilder().buildAutomaton(data, classes, 2, 30);
    data.clear();
    classes.clear();
    generateData(data, classes, predicate, 500000, 10, 50, 2);
    int matchCnt = 0;
    for (int i = 0; i < data.size(); i++) {
      boolean match = automaton.accepts(data.get(i));
      if (match == classes.get(i)) {
        matchCnt++;
      }
    }
    System.out.println("testMod3: precision " + 1.0 * matchCnt / data.size());
    System.out.println(automaton.toString());
  }

  @Test
  public void testNoSubsequentThreeZeroes() {
    final List<TIntList> data = new ArrayList<>();
    final List<Boolean> classes = new ArrayList<>();

    Predicate<TIntList> predicate = word -> {
      for (int i = 0; i < word.size() - 3; i++) {
        boolean matches = true;
        for (int j = 0; j < 3; j++) {
          if (word.get(i + j) != 0) {
            matches = false;
            break;
          }
        }
        if (matches) {
          return false;
        }
      }
      return true;
    };

    generateData(data, classes, predicate, 100000, 10, 50, 2);
    DFA automaton = new IncrementalAutomatonBuilder().buildAutomaton(data, classes, 2, 30);
    data.clear();
    classes.clear();
    generateData(data, classes, predicate, 500000, 10, 50, 2);
    int matchCnt = 0;
    for (int i = 0; i < data.size(); i++) {
      boolean match = automaton.accepts(data.get(i));
      if (match == classes.get(i)) {
        matchCnt++;
      }
    }
    System.out.println("testNoSubsequentThreeZeroes: precision " + 1.0 * matchCnt / data.size());
    System.out.println(automaton.toString());

  }


  @Test
  public void testEndsWith() {
    final List<TIntList> data = new ArrayList<>();
    final List<Boolean> classes = new ArrayList<>();
    final int[] end = {1, 0, 0, 0, 1};

    Predicate<TIntList> predicate = word -> {
      boolean matches = true;
      for (int i = 0; i < end.length; i++) {
        if (word.get(word.size() - end.length + i) != end[i]) {
          matches = false;
          break;
        }
      }
      return matches;
    };

    generateData(data, classes, predicate, 100000, 10, 50, 2);
    DFA automaton = new IncrementalAutomatonBuilder().buildAutomaton(data, classes, 2, 30);
    data.clear();
    classes.clear();
    generateData(data, classes, predicate, 500000, 10, 50, 2);
    int matchCnt = 0;
    for (int i = 0; i < data.size(); i++) {
      boolean match = automaton.accepts(data.get(i));
      if (match == classes.get(i)) {
        matchCnt++;
      }
    }
    System.out.println("testEndsWith: precision " + 1.0 * matchCnt / data.size());
    System.out.println(automaton.toString());

  }

  @Test
  public void testFindSubstring() {
    final List<TIntList> data = new ArrayList<>();
    final List<Boolean> classes = new ArrayList<>();
    final int[] substring = {0, 0, 1, 0, 1};

    Predicate<TIntList> predicate = word -> {
      for (int i = 0; i < word.size() - substring.length + 1; i++) {
        boolean matches = true;
        for (int j = i; j < i + substring.length; j++) {
          if (word.get(j) != substring[j - i]) {
            matches = false;
            break;
          }
        }
        if (matches) {
          return true;
        }
      }
      return false;
    };

    generateData(data, classes, predicate, 100000, 10, 50, 2);
    DFA automaton = new IncrementalAutomatonBuilder().buildAutomaton(data, classes, 2, 30);
    data.clear();
    classes.clear();
    generateData(data, classes, predicate, 500000, 10, 50, 2);
    int matchCnt = 0;
    for (int i = 0; i < data.size(); i++) {
      boolean match = automaton.accepts(data.get(i));
      if (match == classes.get(i)) {
        matchCnt++;
      }
    }
    System.out.println("testFindSubstring: precision " + 1.0 * matchCnt / data.size());
    System.out.println(automaton.toString());
  }

  private void generateData(List<TIntList> data, List<Boolean> classes, Predicate<TIntList> predicate,
                            int sampleCount, int minLen, int maxLen, int alphabetSize) {
    while (data.size() < sampleCount) {
      int len = r.nextInt(maxLen - minLen + 1) + minLen;
      TIntList sample = new TIntArrayList();
      for (int i = 0; i < len; i++) {
        sample.add(r.nextInt(alphabetSize));
      }
      boolean clazz = predicate.test(sample);
      if (clazz && data.size() % 2 == 0) {
        data.add(sample);
        classes.add(true);
      } else if (!clazz && data.size() % 2 == 1) {
        data.add(sample);
        classes.add(false);
      }
    }
  }
}
