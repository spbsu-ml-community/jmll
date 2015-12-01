package com.spbsu.ml.models;

import org.jetbrains.annotations.NotNull;


import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.util.Pair;
import com.spbsu.ml.BFGrid;
import com.spbsu.commons.math.Func;
import com.spbsu.commons.math.Trans;
import com.spbsu.ml.func.Ensemble;
import gnu.trove.map.hash.TObjectDoubleHashMap;
import gnu.trove.procedure.TObjectDoubleProcedure;

import java.util.*;

/**
 * User: solar
 * Date: 28.04.14
 * Time: 8:31
 */
public final class ModelTools {
  private ModelTools() {
  }

  public static class CompiledOTEnsemble extends Func.Stub {
    public static class Entry {
      private final int[] bfIndices;
      private final double value;

      public Entry(final int[] bfIndices, final double value) {
        this.bfIndices = bfIndices;
        this.value = value;
      }

      public int[] getBfIndices() {
        return bfIndices;
      }

      public double getValue() {
        return value;
      }
    }

    private final List<Entry> entries;
    private final BFGrid grid;

    public CompiledOTEnsemble(final List<Entry> entries, final BFGrid grid) {
      this.entries = entries;
      this.grid = grid;
    }

    @Override
    public double value(final Vec x) {
      if (grid == null)
        return 0.;

      final byte[] binary = new byte[grid.rows()];
      grid.binarize(x, binary);
      double result = 0.;
      for (final Entry entry : entries) {
        final int[] bfIndices = entry.getBfIndices();
        double increment = entry.getValue();
        for (int j = 0; j < bfIndices.length; j++) {
          if (!grid.bf(bfIndices[j]).value(binary)) {
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

    public List<Entry> getEntries() {
      return Collections.unmodifiableList(entries);
    }

    public BFGrid getGrid() {
      return grid;
    }
  }

  private static class ConditionEntry{
    public final int[] features;

    private ConditionEntry(final int[] features) {
      this.features = features;
    }

    @Override
    public boolean equals(final Object o) {
      if (this == o) return true;
      if (o == null || getClass() != o.getClass()) return false;

      final ConditionEntry that = (ConditionEntry) o;
      return Arrays.equals(features, that.features);
    }

    @Override
    public int hashCode() {
      return Arrays.hashCode(features);
    }
  }

  public static CompiledOTEnsemble compile(final Ensemble<ObliviousTree> input) {
    final BFGrid grid = input.size() > 0 ? input.models[0].grid() : null;
    final TObjectDoubleHashMap<ConditionEntry> scores = new TObjectDoubleHashMap<>();
    for (int i = 0; i < input.size(); i++) {
      final ObliviousTree model = input.models[i];
      final List<BFGrid.BinaryFeature> features = model.features();
      for (int b = 0; b < model.values().length; b++) {
        final Set<BFGrid.BinaryFeature> currentSet = new HashSet<BFGrid.BinaryFeature>();
        final double[] values = model.values();
        double value = 0;
        final int bitsB = MathTools.bits(b);
        for (int a = 0; a < values.length; a++) {
          final int bitsA = MathTools.bits(a);
          if (MathTools.bits(a & b) >= bitsA)
            value += (((bitsA + bitsB) & 1) > 0 ? -1 : 1) * values[a];
        }
        for (int f = 0; f < features.size(); f++) {
          if ((b >> f & 1) != 0) {
            currentSet.add(features.get(features.size() - f - 1));
          }
        }
        if (value != 0.) {
          final int[] currentSetA = new int[currentSet.size()];
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
    final List<Pair<ConditionEntry, Double>> sortedScores = new ArrayList<>();

    scores.forEachEntry(new TObjectDoubleProcedure<ConditionEntry>() {
      @Override
      public boolean execute(final ConditionEntry a, final double b) {
        sortedScores.add(Pair.create(a, b));
        return true;
      }
    });
    Collections.sort(sortedScores, new Comparator<Pair<ConditionEntry, Double>>() {
      @Override
      public int compare(final Pair<ConditionEntry, Double> o1, final Pair<ConditionEntry, Double> o2) {
        if (o1.first.features.length > o2.first.features.length)
          return 1;
        if (o1.first.features.length == o2.first.features.length) {
          int index = 0;
          while (o1.first.features[index] == o2.first.features[index])
            index++;
          return Integer.compare(o1.first.features[index], o2.first.features[index]);
        }
        return -1;
      }
    });

    final List<CompiledOTEnsemble.Entry> entries = new ArrayList<>(scores.size());
    final int[] freqs = new int[grid.size()];

    for (final Pair<ConditionEntry, Double> score : sortedScores) {
      entries.add(new CompiledOTEnsemble.Entry(score.first.features, score.second));
      for (int i = 0; i < score.first.features.length; i++) {
        freqs[score.first.features[i]]++;
      }
    }

    return new CompiledOTEnsemble(entries, grid);
  }

  @NotNull
  public static Ensemble<ObliviousTree> castEnsembleItems(@NotNull final Ensemble<Trans> model) {
    final ObliviousTree[] trees = new ObliviousTree[model.models.length];
    for (int i = 0; i < model.models.length; i++) {
      trees[i] = (ObliviousTree) model.models[i];
    }
    return new Ensemble<>(trees, model.weights);
  }
}
