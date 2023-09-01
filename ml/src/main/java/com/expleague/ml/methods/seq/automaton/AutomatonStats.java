package com.expleague.ml.methods.seq.automaton;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.seq.Seq;
import com.expleague.commons.seq.regexp.Alphabet;
import com.expleague.ml.TargetFunc;
import com.expleague.ml.data.set.DataSet;
import com.expleague.ml.loss.L2;
import com.expleague.ml.loss.WeightedL2;
import gnu.trove.list.TDoubleList;
import gnu.trove.list.array.TDoubleArrayList;
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
  private final Vec weights;
  private TDoubleList stateSum = new TDoubleArrayList();
  private TDoubleList stateSum2 = new TDoubleArrayList();
  private TDoubleList stateWeight = new TDoubleArrayList();
  private List<TIntSet> samplesViaState = new ArrayList<>();
  private List<TIntIntMap> samplesEndState = new ArrayList<>();

  public AutomatonStats(AutomatonStats<T> other) {
    automaton = other.automaton.copy();
    alphabet = other.alphabet;
    dataSet = other.dataSet;
    target = other.target;
    weights = other.weights;
    samplesEndState = new ArrayList<>(other.samplesEndState);
    samplesViaState = new ArrayList<>(other.samplesViaState);
    stateSum = new TDoubleArrayList(other.stateSum);
    stateSum2 = new TDoubleArrayList(other.stateSum2);
    stateWeight = new TDoubleArrayList(other.stateWeight);
  }

  public AutomatonStats(Alphabet<T> alphabet, DataSet<Seq<T>> dataSet, TargetFunc loss) {
    automaton = new DFA<T>(alphabet);
    this.dataSet = dataSet;
    if (loss instanceof final WeightedL2 weightedLoss) {
      this.weights = weightedLoss.getWeights();
      this.target = weightedLoss.target();
    } else if (loss instanceof L2) {
      this.target = VecTools.copy(((L2) loss).target());
      this.weights = new ArrayVec(target.length());
      VecTools.fill(this.weights, 1);
      VecTools.scale(this.target, this.weights);
    } else {
      throw new IllegalArgumentException();
    }

    this.alphabet = alphabet;
    final TIntSet allIndicesSet = new TIntHashSet();
    final TIntIntMap allIndicesMap = new TIntIntHashMap();

    for (int i = 0; i < dataSet.length(); i++) {
      allIndicesSet.add(i);
      allIndicesMap.put(i, 0);
    }

    stateWeight.add(dataSet.length());
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

  public TDoubleList getStateWeight() {
    return stateWeight;
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

  public void setStateWeight(TDoubleList stateWeight) {
    this.stateWeight = stateWeight;
  }

  public void setAutomaton(DFA<T> automaton) {
    this.automaton = automaton;
  }

  public Vec getWeights() {
    return weights;
  }

  public int addNewState() {
    final int newState = automaton.createNewState();
    samplesViaState.add(new TIntHashSet());
    samplesEndState.add(new TIntIntHashMap());
    stateWeight.add(0);
    stateSum.add(0);
    stateSum2.add(0);

    return newState;
  }

}