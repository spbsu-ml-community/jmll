package automaton;

import com.spbsu.commons.seq.Seq;
import com.spbsu.commons.seq.regexp.Alphabet;
import gnu.trove.list.TIntList;
import gnu.trove.list.array.TIntArrayList;

import java.util.*;

public class DFA<T> {
  private int stateCount = 0;
  private int[][] transitions = new int[0][];
  private final Alphabet<T> alphabet;
  private int[]stateClasses = new int[0];

  public DFA(Alphabet<T> alphabet) {
    this.alphabet = alphabet;
    createNewState();
  }

  public int getStateCount() {
    return stateCount;
  }

  public int createNewState() {
    return createNewState(0);
  }

  public int createNewState(int stateClass) {
    int[][] newTransitions = new int[stateCount + 1][];
    System.arraycopy(transitions, 0, newTransitions, 0, stateCount);
    newTransitions[stateCount] = new int[alphabet.size() + 1];
    Arrays.fill(newTransitions[stateCount], -1);
    transitions = newTransitions;

    int[] newStateClasses = new int[stateCount + 1];
    System.arraycopy(stateClasses, 0, newStateClasses, 0, stateCount);
    stateClasses = newStateClasses;
    stateClasses[stateCount] = stateClass;
    return stateCount++;
  }

  public void removeState(int state) {
    int[][] newTransitions = new int[stateCount - 1][];
    System.arraycopy(transitions, 0, newTransitions, 0, state);
    System.arraycopy(transitions, state + 1, newTransitions, state, stateCount - state - 1);
    transitions = newTransitions;

    int[] newStateClasses = new int[stateCount - 1];
    System.arraycopy(stateClasses, 0, newStateClasses, 0, state);
    System.arraycopy(stateClasses, state + 1, newStateClasses, state, stateCount - state - 1);
    stateClasses = newStateClasses;
    stateCount--;

    for (int i = 0; i < stateCount; i++) {
      for (int j = 0; j < alphabet.size() + 1; j++) {
        if (transitions[i][j] >= state) {
          transitions[i][j]--;
        }
      }
    }
  }

  public int getTransition(int from, T c) {
    return transitions[from][alphabet.index(c)];
  }

  public boolean hasTransition(int from, T c) {
    return transitions[from][alphabet.index(c)] != -1;
  }

  public void addTransition(int from, int to, T c) {
    transitions[from][alphabet.index(c)] = to;
  }

  public void removeTransition(int from, T c) {
    transitions[from][alphabet.index(c)] = -1;
  }

  public void setStateClass(int state, int stateClass) {
    stateClasses[state] = stateClass;
  }

  public int getStateClass(int state) {
    return stateClasses[state];
  }

  public int getWordClass(final Seq<T> word) {
    return stateClasses[run(word)];
  }

  public int getStartState() {
    return 0;
  }

  public int run(final Seq<T> word) {
    return runFrom(0, getStartState(), word);
  }

  public int runFrom(final int startWordPos, final int startState, final Seq<T> word) {
    int curState = startState;
    for (int i = startWordPos; i < word.length(); i++) {
      final int trans = transitions[curState][alphabet.index(word.at(i))];
      if (trans == -1) {
        return curState;
      }
      curState = trans;
    }
    return curState;
  }

  @Override
  public String toString() {
    String result = "";
    for (int i = 0; i < stateCount; i++) {
      for (int c = 0; c < alphabet.size() + 1; c++) {
        if (transitions[i][c] != -1) {
          result += String.format("%d -> %d [label=\"%s\"];", i, transitions[i][c],
                  alphabet.getT(alphabet.get(c)).toString());
        }
      }
    }
    return result;
  }

  public DFA<T> copy() {
    DFA<T> result = new DFA<>(alphabet);
    result.stateCount = stateCount;
    result.stateClasses = Arrays.copyOf(stateClasses, stateCount);
    result.transitions = new int[stateCount][];
    for (int i = 0; i < stateCount; i++) {
      result.transitions[i] = Arrays.copyOf(transitions[i], alphabet.size() + 1);
    }
    return result;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }

    final DFA other = (DFA) o;

    return other.alphabet.equals(alphabet)
            && other.stateCount == stateCount
            && Arrays.deepEquals(other.transitions, transitions)
            && Arrays.equals(other.stateClasses, stateClasses);

  }
}
