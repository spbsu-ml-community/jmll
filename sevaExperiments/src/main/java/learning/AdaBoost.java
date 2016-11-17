package learning;

import automaton.DFA;
import com.spbsu.commons.seq.Seq;
import com.spbsu.commons.seq.regexp.Alphabet;
import gnu.trove.list.TDoubleList;
import gnu.trove.list.TIntList;
import gnu.trove.list.array.TDoubleArrayList;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

public class AdaBoost<T> {
  public Function<Seq<T>, Integer> boost(final List<Seq<T>> data, final TIntList classes, final Alphabet<T> alphabet,
                                         final int classCount, final int automatonIterCount, final int boostIterCount) {
    List<DFA<T>> classifiers = new ArrayList<>();
    TDoubleList weights = new TDoubleArrayList(data.size());
    TDoubleList classifierWeights = new TDoubleArrayList();
    for (int i = 0; i < data.size(); i++) {
      weights.add(1.0 / data.size());
    }

    IncrementalAutomatonBuilder<T> automatonBuilder = new IncrementalAutomatonBuilder<>(new OptimizedCostFunction<>());
    for (int iter = 0; iter < boostIterCount; iter++) {
      DFA<T> automaton = automatonBuilder.buildAutomaton(data, classes, weights,
              alphabet, classCount, automatonIterCount);
      double error = 0;
      for (int i = 0; i < data.size(); i++) {
        if (automaton.getWordClass(data.get(i)) != classes.get(i)) {
          error += weights.get(i);
        }
      }
      final double beta = Math.log((1 - error) / error) + Math.log(classCount - 1);
      classifiers.add(automaton);
      classifierWeights.add(beta);
      for (int i = 0; i < data.size(); i++) {
        if (automaton.getWordClass(data.get(i)) != classes.get(i)) {
          weights.set(i, Math.exp(beta) * weights.get(i));
        }
      }
      final double sumW = weights.sum();
      weights.transformValues(w -> w / sumW);
      System.out.printf("Iteration %d, classifier error %f, accuracy %f\n", iter, error, getAccuracy(data, classes,
              getClassifier(classifiers, classifierWeights, classCount)));
    }
    return getClassifier(classifiers, classifierWeights, classCount);
  }

  private Function<Seq<T>, Integer> getClassifier(final List<DFA<T>> classifiers, final TDoubleList classifierWeights,
                                                  final int classCount) {
    return seq -> {
      final double[] classWeights = new double[classCount];

      for (int i = 0; i < classifiers.size(); i++) {
        final int clazz = classifiers.get(i).getWordClass(seq);
        classWeights[clazz] += 1 / classifierWeights.get(i);
      }
      int result = 0;
      for (int i = 1; i < classCount; i++) {
        if (classWeights[i] > classWeights[result]) {
          result = i;
        }
      }
      return result;
    };
  }

  private double getAccuracy(final List<Seq<T>> data, final TIntList classes,
                             final Function<Seq<T>, Integer> classifier) {
    int matchCnt = 0;
    for (int i = 0; i < data.size(); i++) {
      if (classifier.apply(data.get(i)) == classes.get(i)) {
        matchCnt++;
      }
    }
    return 1.0 * matchCnt / data.size();
  }
}
