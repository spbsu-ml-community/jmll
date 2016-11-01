package automaton;

import gnu.trove.list.TIntList;

import java.util.*;

public class DFA {
  private int stateCount = 0;
  private final List<int[]> transitions = new ArrayList<>();
  private final int alphabetSize;
  private final List<Boolean> isFinalState = new ArrayList<>();

  public DFA(int alphabetSize) {
    this.alphabetSize = alphabetSize;
    createNewState();
  }

  public int getStateCount() {
    return stateCount;
  }

  public int createNewState() {
    int[] trans = new int[alphabetSize];
    for (int i = 0; i < alphabetSize; i++) {
      trans[i] = -1;
    }
    transitions.add(trans);
    isFinalState.add(false);
    return stateCount++;
  }

  public void removeState(int state) {
    transitions.remove(state);
    isFinalState.remove(state);
    stateCount--;
    for (int i = 0; i < stateCount; i++) {
      for (int j = 0; j < alphabetSize; j++) {
        if (transitions.get(i)[j] >= state) {
          transitions.get(i)[j]--;
        }
      }
    }
  }

  public int getTransition(int from, int c) {
    return transitions.get(from)[c];
  }

  public boolean hasTransition(int from, int c) {
    return transitions.get(from)[c] != -1;
  }

  public void addTransition(int from, int to, int c) {
    transitions.get(from)[c] = to;
  }

  public void removeTransition(int from, int c) {
    transitions.get(from)[c] = -1;
  }

  public void markFinalState(int state, boolean isFinal) {
    isFinalState.set(state, isFinal);
  }

  public boolean accepts(final TIntList word) {
    return isFinalState.get(run(word));
  }

  public int getStartState() {
    return 0;
  }

  public boolean isStateFinal(int from) {
    return isFinalState.get(from);
  }

  public int run(final TIntList word) {
    return runFrom(0, getStartState(), word);
  }

  public int runFrom(final int startWordPos, final int startState, final TIntList word) {
    int curState = startState;
    for (int i = startWordPos; i < word.size(); i++) {
      final int trans = transitions.get(curState)[word.get(i)];
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
      for (int c = 0; c < alphabetSize; c++) {
        if (transitions.get(i)[c] != -1) {
          result += String.format("%d -> %d [label=%d];", i, transitions.get(i)[c], c);
        }
      }
    }
    return result;
  }
}
