package learning;

import automaton.DFA;
import com.spbsu.commons.seq.Seq;
import com.spbsu.commons.seq.regexp.Alphabet;
import gnu.trove.list.TDoubleList;
import gnu.trove.list.TIntList;
import gnu.trove.map.TIntIntMap;
import gnu.trove.map.hash.TIntIntHashMap;
import gnu.trove.set.TIntSet;
import gnu.trove.set.hash.TIntHashSet;

import java.util.ArrayList;
import java.util.List;

public class LearningState<T> {
  private DFA<T> automaton;
  private final int classCount;

  private final Alphabet<T> alphabet;
  private final List<Seq<T>> data;
  private final TIntList classes;
  private final TDoubleList weights;
  private List<TIntSet> samplesViaState = new ArrayList<>();
  private List<TIntIntMap> samplesEndState = new ArrayList<>();
  private List<double[]> stateClassWeight = new ArrayList<>();


  public LearningState(LearningState<T> other) {
    automaton = other.automaton.copy();
    classCount = other.classCount;
    alphabet = other.alphabet;
    data = other.data;
    classes = other.classes;
    weights = other.weights;
    samplesEndState = new ArrayList<>(other.samplesEndState);
    samplesViaState = new ArrayList<>(other.samplesViaState);
    stateClassWeight = new ArrayList<>(other.stateClassWeight);
  }

  public LearningState(Alphabet<T> alphabet, int classCount, List<Seq<T>> data, TIntList classes, TDoubleList weights) {
    automaton = new DFA<T>(alphabet);
    this.data = data;
    this.classes = classes;
    this.weights = weights;
    this.classCount = classCount;
    this.alphabet = alphabet;
    final TIntSet allIndicesSet = new TIntHashSet();
    final TIntIntMap allIndicesMap = new TIntIntHashMap();

    stateClassWeight.add(new double[classCount]);

    for (int i = 0; i < data.size(); i++) {
      allIndicesSet.add(i);
      allIndicesMap.put(i, 0);
      stateClassWeight.get(0)[classes.get(i)] += weights.get(i);
    }


    samplesEndState.add(allIndicesMap);
    samplesViaState.add(allIndicesSet);

    samplesEndState.add(new TIntIntHashMap());
    samplesViaState.add(new TIntHashSet());
  }

  public Alphabet<T> getAlphabet() {
    return alphabet;
  }

  public DFA<T> getAutomaton() {
    return automaton;
  }

  public List<TIntSet> getSamplesViaState() {
    return samplesViaState;
  }

  public List<TIntIntMap> getSamplesEndState() {
    return samplesEndState;
  }

  public List<double[]> getStateClassWeight() {
    return stateClassWeight;
  }

  public List<Seq<T>> getData() {
    return data;
  }

  public TIntList getClasses() {
    return classes;
  }

  public TDoubleList getWeights() {
    return weights;
  }

  public void setSamplesViaState(List<TIntSet> samplesViaState) {
    this.samplesViaState = samplesViaState;
  }

  public void setSamplesEndState(List<TIntIntMap> samplesEndState) {
    this.samplesEndState = samplesEndState;
  }

  public void setStateClassWeight(List<double[]> stateClassWeight) {
    this.stateClassWeight = stateClassWeight;
  }

  public int getClassCount() {
    return classCount;
  }

  public void setAutomaton(DFA<T> automaton) {
    this.automaton = automaton;
  }
}