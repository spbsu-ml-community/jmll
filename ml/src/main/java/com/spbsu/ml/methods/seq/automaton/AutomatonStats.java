package com.spbsu.ml.methods.seq.automaton;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.seq.Seq;
import com.spbsu.commons.seq.regexp.Alphabet;
import com.spbsu.ml.data.set.DataSet;
import gnu.trove.list.TDoubleList;
import gnu.trove.list.TIntList;
import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.map.TIntIntMap;
import gnu.trove.map.hash.TIntIntHashMap;
import gnu.trove.set.TIntSet;
import gnu.trove.set.hash.TIntHashSet;

import java.util.ArrayList;
import java.util.List;

public class AutomatonStats<T> {
  private DFA<T> automaton;

  private final Alphabet<T> alphabet;
  private final DataSet<Seq<T>> dataSet;
  private final Vec target;
  private TDoubleList stateSum = new TDoubleArrayList();
  private TDoubleList stateSum2 = new TDoubleArrayList();
  private TIntList stateSize = new TIntArrayList();
  private List<TIntSet> samplesViaState = new ArrayList<>();
  private List<TIntIntMap> samplesEndState = new ArrayList<>();

  public AutomatonStats(AutomatonStats<T> other) {
    automaton = other.automaton.copy();
    alphabet = other.alphabet;
    dataSet = other.dataSet;
    target = other.target;
    samplesEndState = new ArrayList<>(other.samplesEndState);
    samplesViaState = new ArrayList<>(other.samplesViaState);
    stateSum = new TDoubleArrayList(other.stateSum);
    stateSum2 = new TDoubleArrayList(other.stateSum2);
    stateSize = new TIntArrayList(other.stateSize);
  }

  public AutomatonStats(Alphabet<T> alphabet, DataSet<Seq<T>> dataSet, Vec target) {
    automaton = new DFA<T>(alphabet);
    this.dataSet = dataSet;
    this.target = target;
    this.alphabet = alphabet;
    final TIntSet allIndicesSet = new TIntHashSet();
    final TIntIntMap allIndicesMap = new TIntIntHashMap();

    for (int i = 0; i < dataSet.length(); i++) {
      allIndicesSet.add(i);
      allIndicesMap.put(i, 0);
    }

    stateSize.add(dataSet.length());
    stateSum.add(VecTools.sum(target));
    stateSum2.add(VecTools.sum2(target));

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

  public DataSet<Seq<T>> getDataSet() {
    return dataSet;
  }

  public TDoubleList getStateSum() {
    return stateSum;
  }

  public TDoubleList getStateSum2() {
    return stateSum2;
  }

  public void setSamplesViaState(List<TIntSet> samplesViaState) {
    this.samplesViaState = samplesViaState;
  }

  public void setSamplesEndState(List<TIntIntMap> samplesEndState) {
    this.samplesEndState = samplesEndState;
  }

  public TIntList getStateSize() {
    return stateSize;
  }

  public Vec getTarget() {
    return target;
  }

  public void setStateSum(TDoubleList stateSum) {
    this.stateSum = stateSum;
  }

  public void setStateSum2(TDoubleList stateSum2) {
    this.stateSum2 = stateSum2;
  }

  public void setStateSize(TIntList stateSize) {
    this.stateSize = stateSize;
  }

  public void setAutomaton(DFA<T> automaton) {
    this.automaton = automaton;
  }
}