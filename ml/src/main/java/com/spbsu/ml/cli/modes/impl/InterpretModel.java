package com.spbsu.ml.cli.modes.impl;

import com.spbsu.commons.func.Computable;
import com.spbsu.commons.io.StreamTools;
import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.math.Trans;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.commons.util.logging.Logger;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.Binarize;
import com.spbsu.ml.cli.builders.methods.grid.GridBuilder;
import com.spbsu.ml.cli.modes.AbstractMode;
import com.spbsu.ml.data.impl.BinarizedDataSet;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.tools.DataTools;
import com.spbsu.ml.data.tools.Pool;
import com.spbsu.ml.func.Ensemble;
import com.spbsu.ml.io.ModelsSerializationRepository;
import com.spbsu.ml.meta.PoolFeatureMeta;
import com.spbsu.ml.models.ModelTools;
import com.spbsu.ml.models.ObliviousTree;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.map.TObjectIntMap;
import gnu.trove.map.hash.TObjectDoubleHashMap;
import gnu.trove.map.hash.TObjectIntHashMap;
import gnu.trove.set.hash.TIntHashSet;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.MissingArgumentException;

import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.function.IntPredicate;
import java.util.stream.Collectors;

import static com.spbsu.ml.cli.JMLLCLI.*;

/**
 * User: solar
 * Date: 16.05.17
 */
public class InterpretModel extends AbstractMode {
  public void run(final CommandLine command) throws MissingArgumentException, IOException {
    if (!command.hasOption(MODEL_OPTION))
      throw new MissingArgumentException("Please provide 'MODEL_OPTION'");
    if (!command.hasOption(GRID_OPTION))
      throw new MissingArgumentException("Please provide 'GRID_OPTION'");
    if (!command.hasOption(LEARN_OPTION))
      throw new MissingArgumentException("Please provide 'LEARN_OPTION'");
    final Pool<?> pool;
    if (command.hasOption(JSON_FORMAT))
      pool = DataTools.loadFromFile(command.getOptionValue(LEARN_OPTION));
    else
      pool = DataTools.loadFromFeaturesTxt(command.getOptionValue(LEARN_OPTION));

    final BFGrid grid = BFGrid.CONVERTER.convertFrom(StreamTools.readFile(new File(command.getOptionValue(GRID_OPTION))));

    boolean splits = false;
    int topSplits = 100;
    boolean histogram = false;
    boolean mhistogram = false;
    boolean linear = false;
    final TIntArrayList histogramPath = new TIntArrayList();
    final TIntArrayList mhistogramPath = new TIntArrayList();
    if (command.hasOption(INTERPRET_MODE_OPTION)) {
      final String value = command.getOptionValue(INTERPRET_MODE_OPTION);
      final String[] split = value.split("/,/");
      for (final String opt: split) {
        if (opt.startsWith("splits")) {
          splits = true;
          if (opt.length() > "splits()".length())
            topSplits = Integer.parseInt(opt.substring("splits(".length(), opt.length() - 1));
        }
        else if (opt.startsWith("histogram")) {
          histogram = true;
          if (opt.length() > "histogram()".length()) {
            final String features  = opt.substring("histogram(".length(), opt.length() - 1);
            for (final String feature : features.split("/,/")) {
              for (int f = 0; f < grid.rows(); f++) {
                final BFGrid.BFRow row = grid.row(f);
                final String fname = pool.features()[row.origFIndex].id();
                if (feature.startsWith(fname)) {
                  final int bin = Integer.parseInt(feature.substring(fname.length() + 1, feature.length() - 1));
                  histogramPath.add(row.bf(bin).bfIndex);
                  break;
                }
              }
            }

          }
        }
        else if (opt.startsWith("mhistogram")) {
          mhistogram = true;
          if (opt.length() > "mhistogram()".length()) {
            final String features  = opt.substring("histogram(".length(), opt.length() - 1);
            for (final String feature : features.split("/,/")) {
              for (int f = 0; f < grid.rows(); f++) {
                final BFGrid.BFRow row = grid.row(f);
                final String fname = pool.features()[row.origFIndex].id();
                if (feature.startsWith(fname)) {
                  final int bin = Integer.parseInt(feature.substring(fname.length() + 1, feature.length() - 1));
                  mhistogramPath.add(row.bf(bin).bfIndex);
                  break;
                }
              }
            }

          }
        }
        else if (opt.equals("linear")) {
          linear = true;
        }
      }
    }

    final ModelsSerializationRepository serializationRepository;
    final GridBuilder gridBuilder = new GridBuilder();
    gridBuilder.setGrid(grid);
    serializationRepository = new ModelsSerializationRepository(gridBuilder.create());
    try {
      final Computable model = DataTools.readModel(command.getOptionValue(MODEL_OPTION), serializationRepository);
      if (!(model instanceof Ensemble))
        throw new IllegalArgumentException("Provided model is not ensemble");
      final Ensemble ensemble = (Ensemble) model;
      if (ensemble.size() == 0 )
        throw new IllegalArgumentException("Provided ensemble is empty");

      final ArrayList<ObliviousTree> trees = new ArrayList<>();
      for(final Trans component: ensemble.models) {
        if (!(component instanceof ObliviousTree))
          throw new IllegalArgumentException("This component type is not supported: " + component.getClass());
        trees.add((ObliviousTree) component);
      }
      final Ensemble<ObliviousTree> otEnsamble = new Ensemble<>(trees.toArray(new ObliviousTree[trees.size()]), ensemble.weights);
      @SuppressWarnings("unchecked")
      final ModelTools.CompiledOTEnsemble compile = ModelTools.compile(otEnsamble);
      final List<ModelTools.CompiledOTEnsemble.Entry> entries = new ArrayList<>(compile.getEntries());
      TObjectIntMap<ModelTools.CompiledOTEnsemble.Entry> entryCount = new TObjectIntHashMap<>();
      {
        final VecDataSet vds = pool.vecData();
        final BinarizedDataSet bds = vds.cache().cache(Binarize.class, VecDataSet.class).binarize(grid);
        for (ModelTools.CompiledOTEnsemble.Entry entry : entries) {
          int weight = 0;
          for (int i = 0; i < vds.length(); i++) {
            final int[] bfIndices = entry.getBfIndices();
            final int length = bfIndices.length;
            boolean fit = true;
            for (int j = 0; j < length; j++) {
              if (!grid.bf(bfIndices[j]).value(i, bds))
                fit = false;
            }
            if (fit)
              weight++;
          }
          entryCount.put(entry, weight);
        }
      }
      entries.sort((a, b) -> Double.compare(Math.abs(b.getValue() * entryCount.get(b)), Math.abs(a.getValue() * entryCount.get(a))));
      final int[] vfeatures;
      {
        final TIntHashSet valuableFeaturesSet = new TIntHashSet();
        entries.stream().flatMapToInt(s -> Arrays.stream(s.getBfIndices())).forEach(valuableFeaturesSet::add);
        vfeatures = valuableFeaturesSet.toArray();
      }

      if (splits)
        topSplits(pool, grid, entries, vfeatures, topSplits);
      if (histogram)
        histograms(pool, grid, entries, histogramPath);
      if (mhistogram)
        mhistograms(pool, grid, entries, histogramPath);
      if (linear || !(splits || histogram || mhistogram))
        linearComponents(pool, grid, entries, entryCount);
    }
    catch (ClassNotFoundException e) {
      e.printStackTrace();
    }
  }

  private void linearComponents(Pool<?> pool, BFGrid grid, List<ModelTools.CompiledOTEnsemble.Entry> entries, TObjectIntMap<ModelTools.CompiledOTEnsemble.Entry> entryCount) {
    for (final ModelTools.CompiledOTEnsemble.Entry entry : entries) {
      final StringBuilder builder = new StringBuilder();
      builder.append(entryCount.get(entry));
      builder.append("\t");
      builder.append(entry.getValue());
      final int[] bfIndices = entry.getBfIndices();
      builder.append("\t");

      for (int i = 0; i < bfIndices.length; i++) {
        if (i > 0)
          builder.append(", ");
        final BFGrid.BinaryFeature binaryFeature = grid.bf(bfIndices[i]);
        builder.append(pool.features()[binaryFeature.findex].id()).append(" > ").append(ftoa(binaryFeature.condition));
      }
      System.out.println(builder.toString());
    }
  }

  private void histograms(Pool<?> pool, BFGrid grid, List<ModelTools.CompiledOTEnsemble.Entry> entries, TIntArrayList histogramPath) {
    final VecDataSet vds = pool.vecData();
    final BinarizedDataSet bds = vds.cache().cache(Binarize.class, VecDataSet.class).binarize(grid);
    for (int i = 0; i < grid.rows(); i++) {
      final BFGrid.BFRow row = grid.row(i);
      final PoolFeatureMeta meta = pool.features()[row.origFIndex];
      System.out.print(meta.id());
      double total = 0;
      final int[] path = histogramPath.toArray();
      for (int bin = 0; bin < row.size(); bin++) {
        final BFGrid.BinaryFeature binaryFeature = row.bf(bin);
        final List<ModelTools.CompiledOTEnsemble.Entry> vfEntries =
            entries.parallelStream()
                .filter(e -> ArrayTools.supset(e.getBfIndices(), path))
                .filter(e -> ArrayTools.indexOf(binaryFeature.bfIndex, e.getBfIndices()) >= 0)
                .collect(Collectors.toList());
        final double weight = expectedWeight(grid, pool.vecData(), bds, vfEntries);
        total += weight;
        if (Math.abs(weight) > MathTools.EPSILON)
          System.out.print(String.format("\t%d:%.3g:%.4g", bin, row.condition(bin), total));
      }
      System.out.println();
    }
  }

  private void mhistograms(Pool<?> pool, BFGrid grid, List<ModelTools.CompiledOTEnsemble.Entry> entries, TIntArrayList histogramPath) {
    final VecDataSet vds = pool.vecData();
    final BinarizedDataSet bds = vds.cache().cache(Binarize.class, VecDataSet.class).binarize(grid);
    for (int i = 0; i < grid.rows(); i++) {
      final BFGrid.BFRow row = grid.row(i);
      final PoolFeatureMeta meta = pool.features()[row.origFIndex];
      System.out.print(meta.id());
      final int[] path = histogramPath.toArray();
      for (int bin = 0; bin < row.size(); bin++) {
        final BFGrid.BinaryFeature binaryFeature = row.bf(bin);
        final List<ModelTools.CompiledOTEnsemble.Entry> vfEntries =
            entries.parallelStream()
                .filter(e -> ArrayTools.supset(e.getBfIndices(), path))
                .filter(e -> ArrayTools.indexOf(binaryFeature.bfIndex, e.getBfIndices()) >= 0)
                .collect(Collectors.toList());
        final double weight = maxWeight(grid, pool.vecData(), bds, vfEntries);
        if (Math.abs(weight) > MathTools.EPSILON)
          System.out.print(String.format("\t%d:%.3g:%.4g", bin, row.condition(bin), weight));
      }
      System.out.println();
    }
  }

  private void topSplits(Pool<?> pool, BFGrid grid, List<ModelTools.CompiledOTEnsemble.Entry> entries, int[] vfeatures, int topSplits) {
    final VecDataSet vds = pool.vecData();
    final BinarizedDataSet bds = vds.cache().cache(Binarize.class, VecDataSet.class).binarize(grid);
    final TObjectDoubleHashMap<int[]> weights = new TObjectDoubleHashMap<>();
    final List<int[]> splitQueue = new ArrayList<>();
    final List<int[]> split = new ArrayList<>();
    Arrays.stream(vfeatures).mapToObj(vf -> new int[]{vf}).forEach(task -> {
      weights.put(task, 100500);
      splitQueue.add(task);
    });
    for (int i = 0; i < topSplits + vfeatures.length; i++) {
      splitQueue.sort(Comparator.comparingDouble(a -> Math.abs(weights.get(a))));
      final int[] vfset = splitQueue.remove(splitQueue.size() - 1);
      final List<ModelTools.CompiledOTEnsemble.Entry> vfEntries =
              entries.parallelStream()
                  .filter(e -> ArrayTools.supset(e.getBfIndices(), vfset))
                  .collect(Collectors.toList());
      final double value = expectedWeight(grid, vds, bds, vfEntries);
      split.add(vfset);
      weights.put(vfset, value);
      vfEntries.stream().flatMapToInt(vfe -> {
        final TIntArrayList variants = new TIntArrayList(vfset.length);
        for (final int index : vfe.getBfIndices()) {
          if (ArrayTools.indexOf(index, vfset) >= 0)
            continue;
          variants.add(index);
        }
        return Arrays.stream(variants.toArray());
      }).sorted().filter(new IntPredicate() {
        int prev = -1;
        @Override
        public boolean test(int value) {
          boolean result = value != prev;
          prev = value;
          return result;
        }
      }).forEach(idx -> {
        final int[] task = new int[vfset.length + 1];
        System.arraycopy(vfset, 0, task, 0, vfset.length);
        task[vfset.length] = idx;
        weights.put(task, value);
        splitQueue.add(task);
      });
    }
    split.sort((a, b)->Double.compare(Math.abs(weights.get(b)), Math.abs(weights.get(a))));
    for (int[] bfIndices : split) {
      final StringBuilder builder = new StringBuilder();
      builder.append(ftoa(weights.get(bfIndices)));
      builder.append("\t");

      for (int i = 0; i < bfIndices.length; i++) {
        if (i > 0)
          builder.append(", ");
        final BFGrid.BinaryFeature binaryFeature = grid.bf(bfIndices[i]);
        builder.append(pool.features()[binaryFeature.findex].id()).append(" > ").append(ftoa(binaryFeature.condition));
      }
      System.out.println(builder.toString());
    }
  }

  private String ftoa(double v) {
    return String.format(Locale.ENGLISH, "%.2f", v);
  }

  private double expectedWeight(BFGrid grid, VecDataSet vds, BinarizedDataSet bds, List<ModelTools.CompiledOTEnsemble.Entry> vfEntries) {
    double total = 0;
    final int power = vds.length();
    for (int j = 0; j < power; j++) {
      final int finalJ = j;
      final double value = vfEntries.stream()
          .filter(entry -> {
            final int[] bfIndices = entry.getBfIndices();
            final int length = bfIndices.length;
            for (int i = 0; i < length; i++) {
              if (!grid.bf(bfIndices[i]).value(finalJ, bds))
                return false;
            }
            return true;
          })
          .mapToDouble(ModelTools.CompiledOTEnsemble.Entry::getValue)
          .sum();
      total += value;
    }
    total /= power;
    return total;
  }

  private double maxWeight(BFGrid grid, VecDataSet vds, BinarizedDataSet bds, List<ModelTools.CompiledOTEnsemble.Entry> vfEntries) {
    double total = 0;
    final int power = vds.length();
    for (int j = 0; j < power; j++) {
      final int finalJ = j;
      final double value = vfEntries.stream()
          .filter(entry -> {
            final int[] bfIndices = entry.getBfIndices();
            final int length = bfIndices.length;
            for (int i = 0; i < length; i++) {
              if (!grid.bf(bfIndices[i]).value(finalJ, bds))
                return false;
            }
            return true;
          })
          .mapToDouble(ModelTools.CompiledOTEnsemble.Entry::getValue)
          .sum();
      total = Math.max(value, total);
    }
    return total;
  }
}
