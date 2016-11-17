import automaton.DFA;
import com.spbsu.commons.seq.CharSeq;
import com.spbsu.commons.seq.CharSeqArray;
import com.spbsu.commons.seq.CharSeqChar;
import com.spbsu.commons.seq.Seq;
import com.spbsu.commons.seq.regexp.Alphabet;
import com.spbsu.commons.seq.regexp.Matcher;
import gnu.trove.list.TIntList;
import gnu.trove.list.array.TIntArrayList;
import learning.AdaBoost;
import learning.IncrementalAutomatonBuilder;
import learning.OptimizedCostFunction;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;

public class TestSpliceData {
  private static final List<String> CLASSES = Arrays.asList("EI", "IE", "N");
  final List<Seq<Character>> trainData = new ArrayList<>();
  final TIntList trainClasses = new TIntArrayList();

  final List<Seq<Character>> testData = new ArrayList<>();
  final TIntList testClasses = new TIntArrayList();

  @Before
  public void loadData() throws IOException {
    final List<CharSeq> data = new ArrayList<>();
    final TIntList classes = new TIntArrayList();
    final int[] classCount = new int[CLASSES.size()];

    Files.readAllLines(Paths.get("src/test/resources/splice.data.txt")).forEach(line -> {
      final String[] tokens = line.split(",");
      final int clazz = CLASSES.indexOf(tokens[0]);
      if (clazz == -1) {
        throw new IllegalStateException("Unknown class " + tokens[0]);
      }
      classes.add(clazz);
      classCount[clazz]++;
      data.add(new CharSeqArray(tokens[2].trim().toCharArray()));
    });

    final int sampleCount = Arrays.stream(classCount).min().orElse(0);

    for (int clazz = 0; clazz < CLASSES.size(); clazz++) {
      int cnt = 0;
      for (int i = 0; i < data.size(); i++) {
        if (classes.get(i) != clazz) {
          continue;
        }
        if (cnt < sampleCount * 7 / 10) {
          trainClasses.add(classes.get(i));
          trainData.add(data.get(i));
        } else if (cnt < sampleCount){
          testClasses.add(classes.get(i));
          testData.add(data.get(i));
        }
        cnt++;
      }
    }
  }

  @Test
  public void runIncrementalBuilder() {

    final DFA<Character> automaton = new IncrementalAutomatonBuilder<Character>(new OptimizedCostFunction<>())
            .buildAutomaton(trainData, trainClasses, new BioAlphabet(), CLASSES.size(), 60);
    System.out.println(automaton.toString());
    System.out.println("Train accuracy of " + getAccuracy(trainData, trainClasses, automaton::getWordClass));
    System.out.println("Test accuracy of  " + getAccuracy(testData, testClasses, automaton::getWordClass));
  }


  @Test
  public void runAdaBoost() {
    final Function<Seq<Character>, Integer> classifier = new AdaBoost<Character>().boost(trainData, trainClasses, new BioAlphabet(), CLASSES.size(), 60, 20);
    System.out.println("Train accuracy of " + getAccuracy(trainData, trainClasses, classifier));
    System.out.println("Test accuracy of  " + getAccuracy(testData, testClasses, classifier));
  }

  private double getAccuracy(List<Seq<Character>> data, TIntList classes, Function<Seq<Character>, Integer> classifier) {
    int passedCnt = 0;
    for (int i = 0; i < classes.size(); i++) {
      if (classifier.apply(data.get(i)) == classes.get(i)) {
        passedCnt++;
      }
    }
    return 1.0 * passedCnt / data.size();
  }

  private class BioAlphabet implements Alphabet<Character> {
    private static final String ALPHABET = "ACGTN";
    private static final String JUNK_CHARS = "DSR";

    @Override
    public int size() {
      return ALPHABET.length();
    }

    @Override
    public int getOrder(Matcher.Condition<Character> c) {
      if (c == Matcher.Condition.ANY) {
        return ALPHABET.length();
      }

      if (!(c instanceof BioCondition)) {
        throw new IllegalArgumentException("Not a bio condition");
      }
      final int pos = ALPHABET.indexOf(((BioCondition) c).my);
      return pos == -1 ? ALPHABET.length() - 1 : pos;
    }

    @Override
    public Matcher.Condition<Character> get(int i) {
      if (i == ALPHABET.length()) {
        //noinspection unchecked
        return Matcher.Condition.ANY;
      }
      return new BioCondition(ALPHABET.charAt(i));
    }

    @Override
    public Matcher.Condition<Character> getByT(Character i) {
      if (i == '$') {
        //noinspection unchecked
        return Matcher.Condition.ANY;
      }
      return new BioCondition(i);
    }

    @Override
    public Character getT(Matcher.Condition condition) {
      if (condition == Matcher.Condition.ANY) {
        return '$';
      };
      if (!(condition instanceof BioCondition)) {
        throw new IllegalArgumentException("Not a bio condition");
      }
      return ((BioCondition) condition).my;
    }

    @Override
    public int index(Character character) {
      if (character == '$') {
        return ALPHABET.length();
      }
      final int pos = ALPHABET.indexOf(character);
      if (pos != -1) {
        return pos;
      }
      return JUNK_CHARS.indexOf(character);
    }

    @Override
    public int index(Seq<Character> seq, int index) {
      return index(seq.at(index));
    }

    private class BioCondition implements Matcher.Condition<Character> {
      private char my;

      BioCondition(char my) {
        this.my = my;
      }

      @Override
      public boolean is(Character frag) {
        return frag == my;
      }

      @Override
      public boolean equals(Object o) {
        if (o == this) {
          return true;
        }
        if (o == null || !(o instanceof BioCondition)) {
          return false;
        }
        BioCondition c = (BioCondition) o;
        return c.my == my;
      }

      @Override
      public int hashCode() {
        return Character.hashCode(my);
      }
    }
  }
}
