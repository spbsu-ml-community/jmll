package com.expleague.ml.methods.seq.automaton;

import com.expleague.commons.math.vectors.SingleValueVec;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.seq.Seq;
import com.expleague.commons.seq.regexp.Alphabet;

import java.util.Arrays;
import java.util.function.Function;

public class DFA<T> implements Function<Seq<T>, Vec> {
  private int stateCount = 0;
  private int[][] transitions = new int[0][];
  private final Alphabet<T> alphabet;

  public DFA(Alphabet<T> alphabet) {
    this.alphabet = alphabet;
    createNewState();
  }

  @Override
  public Vec apply(Seq<T> argument) {
    return new SingleValueVec(run(argument));
  }

  public int getStateCount() {
    return stateCount;
  }

  public int createNewState() {
    int[][] newTransitions = new int[stateCount + 1][];
    System.arraycopy(transitions, 0, newTransitions, 0, stateCount);
    newTransitions[stateCount] = new int[alphabet.size() + 1];
    Arrays.fill(newTransitions[stateCount], -1);
    transitions = newTransitions;

    return stateCount++;
  }

  public void removeState(int state) {
    int[][] newTransitions = new int[stateCount - 1][];
    System.arraycopy(transitions, 0, newTransitions, 0, state);
    System.arraycopy(transitions, state + 1, newTransitions, state, stateCount - state - 1);
    transitions = newTransitions;

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
            && Arrays.deepEquals(other.transitions, transitions);

  }
}
