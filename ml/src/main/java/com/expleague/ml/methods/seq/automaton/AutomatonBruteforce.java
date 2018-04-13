package com.expleague.ml.methods.seq.automaton;

import com.expleague.commons.math.vectors.SingleValueVec;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.seq.Seq;
import com.expleague.commons.seq.regexp.Alphabet;
import com.expleague.ml.data.set.DataSet;
import com.expleague.ml.loss.L2;
import com.expleague.ml.methods.SeqOptimization;
import com.expleague.ml.methods.seq.automaton.transform.AddTransitionTransform;

import java.util.function.Function;

public class AutomatonBruteforce<T, Loss extends L2> implements SeqOptimization<T, Loss> {
  private final Alphabet<T> alphabet;
  private final Function<AutomatonStats<T>, Double> stateEvaluation;
  private final int maxStateCount;

  public AutomatonBruteforce(final Alphabet<T> alphabet,
                             final Function<AutomatonStats<T>, Double> stateEvaluation,
                             final int maxStateCount) {
    this.alphabet = alphabet;
    this.stateEvaluation = stateEvaluation;
    this.maxStateCount = maxStateCount;
  }

  private AutomatonStats<T> bruteforce(int curState, int curAlpha, final AutomatonStats<T> automatonStats) {
    if (curAlpha == alphabet.size()) {
      curAlpha = 0;
      curState++;
    }

    if (curState == maxStateCount) {
      return automatonStats;
    }

    double optCost = Double.MAX_VALUE;
    AutomatonStats<T> optStats = automatonStats;

    for (int toState = 0; toState < maxStateCount; toState++) {
      final AutomatonStats<T> newAutomatonStats = new AddTransitionTransform<>(
              curState, toState, alphabet.getT(alphabet.condition(curAlpha))
      ).applyTransform(automatonStats);

      final AutomatonStats<T> curAutomatonStats = bruteforce(curState, curAlpha + 1, newAutomatonStats);
      final double curCost = stateEvaluation.apply(curAutomatonStats);

      if (curCost < optCost) {
        optCost = curCost;
        optStats = curAutomatonStats;
      }
    }

    return optStats;
  }

  @Override
  public Function<Seq<T>, Vec> fit(DataSet<Seq<T>> learn, Loss loss) {
    AutomatonStats<T> automatonStats = new AutomatonStats<T>(alphabet, learn, loss);
    for (int i = 1; i < maxStateCount; i++) {
      automatonStats.addNewState();
    }

    automatonStats = bruteforce(0, 0, automatonStats);
    final DFA<T> automaton = automatonStats.getAutomaton();

    final double[] stateValue = new double[automaton.getStateCount()];
    for (int i = 0; i < automaton.getStateCount(); i++) {
      stateValue[i] = automatonStats.getStateSum().get(i) / automatonStats.getStateWeight().get(i);
    }
    System.out.println("Cur cost = " + stateEvaluation.apply(automatonStats));
    return argument -> new SingleValueVec(stateValue[automaton.run(argument)]);
  }
}
