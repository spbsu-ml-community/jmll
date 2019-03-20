package com.expleague.ml.embedding.decomp;

import com.expleague.commons.math.MathTools;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.random.FastRandom;
import com.expleague.commons.seq.CharSeq;
import com.expleague.commons.seq.LongSeq;
import com.expleague.commons.util.ArrayTools;
import com.expleague.commons.util.MultiMap;
import com.expleague.commons.util.logging.Interval;
import com.expleague.ml.embedding.Embedding;
import com.expleague.ml.embedding.impl.EmbeddingBuilderBase;
import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.list.array.TLongArrayList;
import gnu.trove.procedure.TLongProcedure;
import gnu.trove.set.TIntSet;
import gnu.trove.set.hash.TIntHashSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

public class MultiDecompBuilder extends EmbeddingBuilderBase {
  private static final Logger log = LoggerFactory.getLogger(MultiDecompBuilder.class);
  private double xMax = 10;
  private double alpha = 0.75;
  private int symDim = 50;
  private int skewDim = 10;
  private final double minimumNorm = 1;

  private FastRandom rng = new FastRandom();

  public MultiDecompBuilder xMax(int xMax) {
    this.xMax = xMax;
    return this;
  }

  public MultiDecompBuilder alpha(double alpha) {
    this.alpha = alpha;
    return this;
  }

  public MultiDecompBuilder dimSym(int dim) {
    this.symDim = dim;
    return this;
  }

  public MultiDecompBuilder dimSkew(int dim) {
    this.skewDim = dim;
    return this;
  }

  public MultiDecompBuilder seed(long seed) {
    rng = new FastRandom(seed);
    return this;
  }

  private double weightingFunc(double x) {
    return x < xMax ? Math.pow((x / xMax), alpha) : 1;
  }

  @Override
  protected boolean isCoocNecessery() {
    return true;
  }

  @Override
  public Embedding<CharSeq> fit() {
    final int size = dict().size();
    final List<Vec> symDecomp = new ArrayList<>();
    final List<Vec> softMaxSym = new ArrayList<>();

    final List<Vec> skewsymDecomp = new ArrayList<>();
    final List<Vec> softMaxSkewsym = new ArrayList<>();

    final TDoubleArrayList bias = new TDoubleArrayList(size);
    final TDoubleArrayList softMaxBias = new TDoubleArrayList(size);
    for (int i = 0; i < size; i++) {
      symDecomp.add(new ArrayVec(IntStream.range(0, symDim).mapToDouble(d -> initializeValue(symDim)).toArray()));
      softMaxSym.add(VecTools.fill(new ArrayVec(symDim), 1.));

      skewsymDecomp.add(new ArrayVec(IntStream.range(0, skewDim).mapToDouble(d -> initializeValue(skewDim)).toArray()));
      softMaxSkewsym.add(VecTools.fill(new ArrayVec(skewDim), 1.));

      bias.add(initializeValue(symDim));
      softMaxBias.add(1);
    }

    final TIntArrayList order = new TIntArrayList(IntStream.range(0, size).toArray());
    rng = new FastRandom();
    for (int iter = 0; iter < T(); iter++) {
      Interval.start();
      order.shuffle(rng);

      final ScoreCalculator scoreCalculator = new ScoreCalculator(size);
      int finalIter = iter;
      ThreadLocal<TLongArrayList> validPairsHolder = ThreadLocal.withInitial(TLongArrayList::new);
      IntStream.range(0, size).parallel().map(order::get).forEach(i -> {
        final Vec sym_i = symDecomp.get(i);
        final Vec skew_i = skewsymDecomp.get(i);
        final Vec softMaxSym_i = softMaxSym.get(i);
        final Vec softMaxSkew_i = softMaxSkewsym.get(i);
        final LongSeq cooc = cooc(i);
        if (cooc.length() < 10000 && cooc.length() > 1000 && finalIter > 0 && VecTools.norm(sym_i) > minimumNorm) {
          final TLongArrayList validPairs = validPairsHolder.get();
          validPairs.reset();
          double qualityThreshold = 0.7;//Math.cos(Math.PI / 2.5);

//          for (int u = 0; u < cooc.length(); u++) {
//            Vec vecA = symDecomp.get(u);
//            qualityThreshold += VecTools.cosine(sym_i, vecA);
//          }
//          qualityThreshold /= cooc.length();
//          qualityThreshold = Math.max(0, qualityThreshold);
          double[] counters = new double[cooc.length()];
          Arrays.fill(counters, Double.NaN);
          for (int u = 0; u < cooc.length(); u++) {
            final Vec vecU = symDecomp.get(u);
            final double norm_u = VecTools.norm(vecU);
            if (norm_u < minimumNorm)
              continue;
            counters[u] = unpackWeight(cooc, u);
            for (int v = u + 1; v < cooc.length(); v++) {
              final Vec vecV = symDecomp.get(v);
              final double norm_v = VecTools.norm(vecV);
              if (norm_v < minimumNorm)
                continue;
              if (VecTools.multiply(vecU, vecV) / norm_u / norm_v > qualityThreshold) {
                validPairs.add(((long) (u + 1) << 32) | (v + 1));
                validPairs.add(((long) (v + 1) << 32) | (u + 1));
              }
            }
          }
          validPairs.sort();
          validPairs.forEach(p -> {
            final int u = (int) (p >>> 32) - 1;
            final int v = (int) (p & 0xFFFFFFFFL) - 1;
            counters[u] += unpackWeight(cooc, v);
            return true;
          });
          List<TIntHashSet> clusters = new ArrayList<>();
          List<List<String>> wordClusters = new ArrayList<>();
          while (true) {
            int max = ArrayTools.max(counters);
            if (max < 0)
              break;
            counters[max] = Double.NaN;
            final TIntHashSet cluster = new TIntHashSet();
            final List<String> wordsCluster = new ArrayList<>();
            cluster.add(max);
            wordsCluster.add(dict().get(max).toString());
            { // form cluster
              int index = -validPairs.binarySearch((long) (max + 1) << 32) - 1;
              long limit = ((long) (max + 2) << 32);
              long p;
              while (index < validPairs.size() && (p = validPairs.getQuick(index)) < limit) {
                int v = (int) (p & 0xFFFFFFFFL) - 1;
                if (!Double.isNaN(counters[v])) {
                  counters[v] = Double.NaN;
                  cluster.add(v);
                  wordsCluster.add(dict().get(unpackB(cooc, v)).toString());
                }
                index++;
              }
            }
            validPairs.forEach(new TLongProcedure() {
              int current = 0;
              float currentWeight;
              @Override
              public boolean execute(long p) { // update counters
                int u = (int) (p >>> 32) - 1;
                int v = (int) (p & 0xFFFFFFFFL) - 1;
                if (u != current) {
                  current = u;
                  currentWeight = cluster.contains(u) ? unpackWeight(cooc, u) : 0.f;
                }
                if (currentWeight != 0f)
                  counters[v] -= currentWeight;
                return true;
              }
            });
            clusters.add(cluster);
            if (cluster.size() == 1)
              continue;

            wordClusters.add(wordsCluster);
          }
          CharSeq word = dict().get(i);
          if (word.equals("apple") || word.equals("lock")) {
            clusters.size();
          }
          if (word.equals("apple") || wordClusters.size() > 1 && wordClusters.get(0).size() / (double)wordClusters.get(1).size() < 10 && wordClusters.get(1).size() > 10) {
            StringBuilder builder = new StringBuilder();
            builder.append(word).append('\n');
            for (List<String> cluster : wordClusters) {
              builder.append('\t').append(cluster.size()).append('\t');
              for (int j = 0; j < cluster.size() && j < 10; j++) {
                builder.append(cluster.get(j)).append(',');
              }
              builder.append('\n');
            }
            System.out.println(builder);
          }
        }
        cooc(i, (j, X_ij) -> {
          final Vec sym_j = symDecomp.get(j);
          final Vec skew_j = skewsymDecomp.get(j);
          final Vec softMaxSym_j = softMaxSym.get(j);
          final Vec softMaxSkew_j = softMaxSkewsym.get(j);
          final double b_i = bias.get(i);
          final double b_j = bias.get(j);

          double asum = VecTools.multiply(sym_i, sym_j);
          double bsum = VecTools.multiply(skew_i, skew_j);
          final int sign = i > j ? -1 : 1;
          final double minfo = Math.log(X_ij);
          final double diff = b_i + b_j + asum + sign * bsum - minfo;
          final double weight = weightingFunc(X_ij);
          final double biasStep = weight * diff;
          scoreCalculator.adjust(i, j, weight, 0.5 * weight * MathTools.sqr(diff));

          update(sym_i, softMaxSym_i, sym_j, softMaxSym_j, diff * weight);
          update(skew_i, softMaxSkew_i, skew_j, softMaxSkew_j, diff * weight * sign);

          bias.setQuick(i, b_i - step() * biasStep / Math.sqrt(softMaxBias.get(i)));
          softMaxBias.setQuick(i, softMaxBias.getQuick(j) + biasStep * biasStep);
          bias.setQuick(j, b_j -step() * biasStep / Math.sqrt(softMaxBias.get(j)));
          softMaxBias.setQuick(j, softMaxBias.getQuick(j) + biasStep * biasStep);
        });
      });

      Interval.stopAndPrint("Iteration: " + iter + " Score: " + scoreCalculator.gloveScore());
    }

    final MultiMap<CharSeq, Vec> mapping = new MultiMap<>();
    for (int i = 0; i < dict().size(); i++) {
      final CharSeq word = dict().get(i);
      mapping.put(word, symDecomp.get(i));
    }

    return null;
  }

  private void update(Vec x_i, Vec softMaxD_i, Vec x_j, Vec softMaxD_j, double step) {
    IntStream.range(0, x_i.dim()).forEach(id -> {
      final double dx_i = x_j.get(id) * step;
      final double dx_j = x_i.get(id) * step;
      final double maxL_i = softMaxD_i.get(id);
      final double maxL_j = softMaxD_j.get(id);
      x_i.adjust(id, -step() * dx_i / Math.sqrt(maxL_i));
      x_j.adjust(id, -step() * dx_j / Math.sqrt(maxL_j));
      softMaxD_i.set(id, maxL_i + MathTools.sqr(dx_i));
      softMaxD_j.set(id, maxL_j + MathTools.sqr(dx_j));
    });
  }

  private synchronized void split(int i, int[] indices) {
    CharSeq word = dict().get(i);
    log.info("Splitting word: " + word);
    final int newIndex = dict().size();
    final LongSeq line = cooc(i);
    final TIntSet removeSet = new TIntHashSet(indices);
    cooc(newIndex, line.sub(indices));
    dict().add(word);
    cooc(i, new LongSeq(line.stream().filter(pack -> removeSet.contains((int) (pack >>> 32))).toArray()));
    for (int index : indices) {
      LongSeq cooc = cooc(index);
      for (int j = 0; j < cooc.length(); j++) {
        long entry = cooc.longAt(j);
        if (entry >>> 32 == i) {
          cooc.data()[j] = (entry & 0x00000000FFFFFFFFL) | ((long)newIndex << 32);
        }
      }
    }
  }

  private double initializeValue(int dim) {
    return (Math.random() - 0.5) / dim;
  }
}
