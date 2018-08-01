package com.expleague.ml.models;

import com.expleague.commons.math.Func;
import com.expleague.commons.math.MathTools;
import com.expleague.commons.math.Trans;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.util.Pair;
import com.expleague.ml.Vectorization;
import com.expleague.ml.func.Ensemble;
import com.expleague.ml.BFGrid;
import com.expleague.ml.impl.BinaryFeatureImpl;
import com.expleague.ml.meta.DSItem;
import com.expleague.ml.meta.FeatureMeta;
import gnu.trove.iterator.TIntIterator;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.map.hash.TObjectDoubleHashMap;
import org.jetbrains.annotations.NotNull;

import java.util.*;
import java.util.function.Function;
import java.util.stream.Stream;

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
      grid.binarizeTo(x, binary);
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

  public static CompiledOTEnsemble compile(final Ensemble<ObliviousTree> ensemble) {
    final BFGrid grid = ensemble.size() > 0 ? ensemble.models[0].grid() : null;
    final TObjectDoubleHashMap<ConditionEntry> scores = new TObjectDoubleHashMap<>();

    for (int treeIndex = 0; treeIndex < ensemble.size(); treeIndex++) {
      final ObliviousTree tree = ensemble.models[treeIndex];

      final List<BFGrid.Feature> features = tree.features();
      final double[] values = tree.values();

      for (int b = 0; b < tree.values().length; b++) {
        final Set<BFGrid.Feature> currentSet = new HashSet<>();

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
          final TIntArrayList conditions = new TIntArrayList(currentSet.size());
          for (BFGrid.Feature aCurrentSet : currentSet) {
            conditions.add(aCurrentSet.index());
          }
          conditions.sort();
          if (grid != null) { // minimize
            final TIntIterator iterator = conditions.iterator();
            BFGrid.Feature prev = null;
            while (iterator.hasNext()) {
              final BFGrid.Feature next = grid.bf(iterator.next());
              if (prev != null && prev.findex() == next.findex()) {
                iterator.remove();
              }
              prev = next;
            }
          }

          final double addedValue = ensemble.weights.get(treeIndex) * value;
          scores.adjustOrPutValue(new ConditionEntry(conditions.toArray()), addedValue, addedValue);
        }
      }
    }

    final List<Pair<ConditionEntry, Double>> sortedScores = new ArrayList<>();
    scores.forEachEntry((entry, value) -> {
      if (Math.abs(value) > 0)
        sortedScores.add(Pair.create(entry, value));
      return true;
    });
    sortedScores.sort((o1, o2) -> {
      if (o1.first.features.length > o2.first.features.length)
        return 1;
      if (o1.first.features.length == o2.first.features.length) {
        int index = 0;
        while (o1.first.features[index] == o2.first.features[index])
          index++;
        return Integer.compare(o1.first.features[index], o2.first.features[index]);
      }
      return -1;
    });

    final List<CompiledOTEnsemble.Entry> entries = new ArrayList<>(scores.size());
//    final int[] freqs = new int[grid.size()];
//
    for (final Pair<ConditionEntry, Double> score : sortedScores) {
      entries.add(new CompiledOTEnsemble.Entry(score.first.features, score.second));
//      for (int i = 0; i < score.first.features.length; i++) {
//        freqs[score.first.features[i]]++;
//      }
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
