package com.spbsu.ml.models;

import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.util.Pair;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.Func;
import com.spbsu.ml.func.Ensemble;
import com.spbsu.ml.models.pgm.ProbabilisticGraphicalModel;
import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.map.hash.TObjectDoubleHashMap;
import gnu.trove.procedure.TObjectDoubleProcedure;
import gnu.trove.set.TIntSet;

import java.util.*;

/**
 * User: solar
 * Date: 28.04.14
 * Time: 8:31
 */
public class ModelTools {
  private static class ConditionEntry{
    public final int[] features;

    private ConditionEntry(int[] features) {
      this.features = features;
    }


    @Override
    public boolean equals(Object o) {
      if (this == o) return true;
      if (o == null || getClass() != o.getClass()) return false;

      ConditionEntry that = (ConditionEntry) o;
      return Arrays.equals(features, that.features);
    }

    @Override
    public int hashCode() {
      return Arrays.hashCode(features);
    }
  }
  public static Func compile(Ensemble<ObliviousTree> input) {
    final BFGrid grid = input.size() > 0 ? input.models[0].grid() : null;
    final TObjectDoubleHashMap<ConditionEntry> scores = new TObjectDoubleHashMap<ConditionEntry>();
    for (int i = 0; i < input.size(); i++) {
      final ObliviousTree model = input.models[i];
      final List<BFGrid.BinaryFeature> features = model.features();
      for (int b = 0; b < model.values().length; b++) {
        final Set<BFGrid.BinaryFeature> currentSet = new HashSet<BFGrid.BinaryFeature>();
        final double[] values = model.values();
        double value = 0;
        int bitsB = MathTools.bits(b);
        for (int a = 0; a < values.length; a++) {
          int bitsA = MathTools.bits(a);
          if (MathTools.bits(a & b) >= bitsA)
            value += (((bitsA + bitsB) & 1) > 0 ? -1 : 1) * values[a];
        }
        for (int f = 0; f < features.size(); f++) {
          if ((b >> f & 1) != 0) {
            currentSet.add(features.get(features.size() - f - 1));
          }
        }
        if (value != 0.) {
          int[] currentSetA = new int[currentSet.size()];
          final Iterator<BFGrid.BinaryFeature> it = currentSet.iterator();
          for (int j = 0; j < currentSetA.length; j++) {
            currentSetA[j] = it.next().bfIndex;
          }
          Arrays.sort(currentSetA);
          final ConditionEntry entry = new ConditionEntry(currentSetA);
          if (scores.containsKey(entry))
            scores.adjustValue(entry, input.weights.get(i) * value);
          else
            scores.put(entry, input.weights.get(i) * value);
        }
      }
    }
    final List<Pair<ConditionEntry, Double>> sortedScores = new ArrayList<Pair<ConditionEntry, Double>>();

    scores.forEachEntry(new TObjectDoubleProcedure<ConditionEntry>() {
      @Override
      public boolean execute(ConditionEntry a, double b) {
        sortedScores.add(Pair.create(a, b));
        return true;
      }
    });
    Collections.sort(sortedScores, new Comparator<Pair<ConditionEntry, Double>>() {
      @Override
      public int compare(Pair<ConditionEntry, Double> o1, Pair<ConditionEntry, Double> o2) {
        if (o1.first.features.length > o2.first.features.length)
          return 1;
        else if (o1.first.features.length == o2.first.features.length) {
          int index = 0;
          while (o1.first.features[index] == o2.first.features[index])
            index++;
          return Integer.compare(o1.first.features[index], o2.first.features[index]);
        }

        return -1;
      }
    });

    final List<int[]> entries = new ArrayList<int[]>(scores.size());
    final TDoubleArrayList values = new TDoubleArrayList(scores.size());
    int[] freqs = new int[grid.size()];

    for (Pair<ConditionEntry, Double> score : sortedScores) {
      entries.add(score.first.features);
      for (int i = 0; i < score.first.features.length; i++) {
        freqs[score.first.features[i]]++;
      }
      values.add(score.second);
    }
    return new Func.Stub() {
      byte[] binary = new byte[grid.rows()];
      @Override
      public synchronized double value(Vec x) {
        if (grid == null)
          return 0.;

        grid.binarize(x, binary);
        double result = 0.;
        for (int i = 0; i < entries.size(); i++) {
          final int[] conditions = entries.get(i);
          double increment = values.get(i);
          for (int j = 0; j < conditions.length; j++) {
            if (!grid.bf(conditions[j]).value(binary)) {
              increment = 0;
              break;
            }
          }
          result += increment;
        }
        return result;
      }

      @Override
      public int dim() {
        return grid.rows();
      }
    };
  }
}
