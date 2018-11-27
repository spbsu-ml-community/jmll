package com.expleague.joom.tags;

import com.expleague.commons.csv.CsvTools;
import com.expleague.commons.math.MathTools;
import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.random.FastRandom;
import com.expleague.commons.seq.CharSeq;
import com.expleague.commons.seq.CharSeqBuilder;
import com.expleague.commons.seq.CharSeqTools;
import com.expleague.commons.util.ArrayTools;
import com.expleague.commons.util.Holder;
import com.expleague.commons.util.Pair;
import com.expleague.commons.util.logging.Interval;
import gnu.trove.map.hash.TIntIntHashMap;
import gnu.trove.map.hash.TObjectIntHashMap;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;
import java.util.stream.IntStream;
import java.util.zip.GZIPInputStream;

public class GloveLikeEmbedding {
  private static final int TRAINING_ITERS = 100;
  private static final double TRAINING_STEP_COEFF = 1e-2;
  private static final FastRandom rng = new FastRandom();
  public static final int DIM = 50;

  private double train(Map<int[], TIntIntHashMap> cooccurrences, Mx leftVectors, Mx rightVectors) {
    final Vec biases = new ArrayVec(rightVectors.rows() + leftVectors.rows());
    VecTools.fillUniformPlus(biases, rng, 1e-3);

    final Mx softMaxLeft = new VecBasedMx(leftVectors.rows(), leftVectors.columns());
    final Mx softMaxRight = new VecBasedMx(rightVectors.rows(), rightVectors.columns());
    final Vec softMaxBias = new ArrayVec(rightVectors.rows() + leftVectors.rows());

    VecTools.fill(softMaxLeft, 1);
    VecTools.fill(softMaxRight, 1);
    VecTools.fill(softMaxBias, 1);

    double score = Double.NEGATIVE_INFINITY;
    for (int iter = 0; iter < TRAINING_ITERS; iter++) {
      Interval.start();
      final double[] scoreArr = new double[]{0, 0, 0};
      cooccurrences.entrySet().parallelStream().forEach(entry -> {
        final int[] generators = entry.getKey();
        final TIntIntHashMap freqs = entry.getValue();
        final Vec totalLeft = new ArrayVec(leftVectors.columns());
        final double[] scoreLocal = new double[]{0, 0, 0};
        freqs.forEachEntry((j, X_ij) -> {
          final Vec right = rightVectors.row(j);
          final double asum = IntStream.of(generators).mapToDouble(i -> VecTools.multiply(leftVectors.row(i), right)).sum();
          final double bias = IntStream.of(generators).mapToDouble(biases::get).sum() + biases.get(j + leftVectors.rows());
          final double diff = bias + asum - Math.log(1 + X_ij);
          final double weight = weightingFunc(1 + X_ij);
          final double v = weight * diff * diff;
          scoreLocal[0] += v;
          scoreLocal[1] += weight;
          scoreLocal[2]++;
          if (Double.isNaN(v))
            System.out.println();
          VecTools.fill(totalLeft, 0.);
          IntStream.of(generators).forEach(i -> { // generators update
            final Vec left = leftVectors.row(i);
            VecTools.append(totalLeft, left);
            final Vec softMax = softMaxLeft.row(i);
            IntStream.range(0, left.dim()).forEach(id -> {
              final double d = TRAINING_STEP_COEFF * weight * diff * right.get(id);
              left.adjust(id, -d / Math.sqrt(softMax.get(id)));
              softMax.adjust(id, d * d);
            });
            final double biasStep = TRAINING_STEP_COEFF * weight * diff;// * biases.get(i);
            biases.adjust(i, -biasStep / Math.sqrt(softMaxBias.get(i)));
            softMaxBias.adjust(i, MathTools.sqr(biasStep));
          });

          { // generated update
            final Vec softMax = softMaxRight.row(j);
            IntStream.range(0, right.dim()).forEach(id -> {
              final double d = TRAINING_STEP_COEFF * weight * diff * totalLeft.get(id);
              right.adjust(id, -d / Math.sqrt(softMax.get(id)));
              softMax.adjust(id, d * d);
            });

            final double biasStep = TRAINING_STEP_COEFF * weight * diff;// * biases.get(j);
            biases.adjust(j + leftVectors.rows(), -biasStep / Math.sqrt(softMaxBias.get(j + leftVectors.rows())));
            softMaxBias.adjust(j + leftVectors.rows(), MathTools.sqr(biasStep));
          }
          return true;
        });
        synchronized (scoreArr) {
          ArrayTools.add(scoreArr, 0, scoreLocal, 0, scoreArr.length);
        }
      });
      score = scoreArr[0] / scoreArr[2];
      Interval.stopAndPrint("Iteration: " + iter + " score: " + score);
    }

    return score;
  }

  private void saveModel(List<CharSeq> ids, Mx vecs, Path to) {
    try (final BufferedWriter writer = Files.newBufferedWriter(to)) {
      for (int i = 0; i < ids.size(); i++) {
        writer.append(ids.get(i)).append('\t').append(MathTools.CONVERSION.convert(vecs.row(i), CharSequence.class));
        writer.newLine();
      }
    }
    catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  private double weightingFunc(double x) {
    return x < 100 ? Math.pow((x / 100), 0.75) : 1;
  }

  public static void main(String[] args) throws Exception {
    final Map<int[], TIntIntHashMap> cooccurrences = new HashMap<>();
    final List<CharSeq> tags = new ArrayList<>();
    final List<CharSeq> subs = new ArrayList<>();
    final TIntIntHashMap tagsFreq = new TIntIntHashMap();
    final TObjectIntHashMap<CharSeq> invTags = new TObjectIntHashMap<>(100000, 0.8f, -1);
    final TObjectIntHashMap<CharSeq> invSubs = new TObjectIntHashMap<>(100000, 0.8f, -1);


    try (final InputStreamReader reader = new InputStreamReader(new GZIPInputStream(Files.newInputStream(Paths.get("/Users/solar/data/joom/searches/search_sessions_tags.csv.gz"))),
        StandardCharsets.UTF_8)) {
      final List<Pair<int[], Long>> currentQuery = new ArrayList<>();
      final Holder<CharSeq> currentUser = new Holder<>();
      final long[] counter = new long[]{0};
      CsvTools.csvLines(reader, ',', '"', '\\', true).limit(200_000_000).forEach(row -> {
        if (++counter[0] % 1000000 == 0)
          System.out.print("\r" + counter[0] + " lines processed");
        final CharSeq user = row.at("user");
        if (!user.equals(currentUser.getValue())) {
          currentQuery.clear();
          currentUser.setValue(user);
        }
        final CharSeq query = row.at("query");
        final CharSeq tag = row.at("tag");
        if (query != null) {
          currentQuery.add(Pair.create(CharSeqTools.split(query, " ", false).map(CharSeqTools::toLowerCase).mapToInt(part -> {
            int subIdx = invSubs.get(part);
            if (subIdx > 0)
              return subIdx;
            final CharSeq partIntern = CharSeq.intern(part);
            invSubs.put(partIntern, subIdx = subs.size());
            subs.add(partIntern);
            return subIdx;
          }).toArray(), row.asLong("ts")));
        }
        else if (tag != null) {
          final long time = row.asLong("ts");
          int tagIdx = invTags.get(tag);
          if (tagIdx < 0) {
            final CharSeq tagIntern = CharSeq.intern(tag);
            invTags.put(tagIntern, tagIdx = tags.size());
            tags.add(tagIntern);
          }
          tagsFreq.adjustOrPutValue(tagIdx, 1, 1);

          final Iterator<Pair<int[], Long>> it = currentQuery.iterator();
          while (it.hasNext()) {
            Pair<int[], Long> next = it.next();
            if (TimeUnit.MILLISECONDS.toMinutes(time - next.second) > 30) {
              it.remove();
              continue;
            }
            final TIntIntHashMap map = cooccurrences.computeIfAbsent(next.first, (key) -> new TIntIntHashMap(1, 0.9f));
            map.adjustOrPutValue(tagIdx, 1, 1);
          }
        }
      });
    }
    System.out.println();

    final List<CharSeq> reducedTags = new ArrayList<>();
    final int[] substitution = new int[tags.size()];
    for (int i = 0; i < tags.size(); i++) {
      if (tagsFreq.get(i) > 50) {
        substitution[i] = reducedTags.size();
        reducedTags.add(tags.get(i));
      }
      else substitution[i] = -1;
    }
    tags.clear();
    System.gc();

    for (int[] query : new ArrayList<>(cooccurrences.keySet())) {
      final TIntIntHashMap substitutedFreqs = new TIntIntHashMap(1, 0.9f);
      final TIntIntHashMap origFreqs = cooccurrences.get(query);
      origFreqs.forEachEntry((tag, freq) -> {
        final int subst = substitution[tag];
        if (subst > 0)
          substitutedFreqs.put(subst, freq);
        return true;
      });
      origFreqs.clear();
      origFreqs.compact();
      cooccurrences.put(query, substitutedFreqs);
    }

    final GloveLikeEmbedding embedding = new GloveLikeEmbedding();
    Mx subsVec = new VecBasedMx(subs.size(), GloveLikeEmbedding.DIM);
    Mx tagsVec = new VecBasedMx(reducedTags.size(), GloveLikeEmbedding.DIM);

    VecTools.fillUniformPlus(subsVec, rng, 1e-3);
    VecTools.fillUniformPlus(tagsVec, rng, 1e-3);

    System.out.println("Starting optimization of " + reducedTags.size() + " tags for " + subs.size() + " words");

    embedding.train(cooccurrences, subsVec, tagsVec);
    embedding.saveModel(reducedTags, tagsVec, Paths.get("/Users/solar/data/joom/searches/tags-vec-" + DIM + ".txt"));
    embedding.saveModel(subs, subsVec, Paths.get("/Users/solar/data/joom/searches/subs-vec-" + DIM + ".txt"));
  }

  private static void visitVariants(CharSeq query, Consumer<CharSeq> variantConsumer) {
    final CharSeq[] parts = CharSeqTools.split(query, " ", false).map(CharSeqTools::toLowerCase).map(CharSeq::create).toArray(CharSeq[]::new);
    for (int i = 0; i < 1 << parts.length; i++) {
      final CharSeqBuilder builder = new CharSeqBuilder(parts.length);
      int mask = i;
      for (int j = 0; j < parts.length; j++, mask >>= 1) {
        if ((mask & 1) > 0)
          builder.append(parts[j]);
      }
      variantConsumer.accept(builder.build());
    }
  }
}
