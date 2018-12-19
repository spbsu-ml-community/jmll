package com.expleague.joom.tags;

import com.expleague.commons.csv.CsvRow;
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
import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.map.hash.TIntFloatHashMap;
import gnu.trove.map.hash.TIntIntHashMap;
import gnu.trove.map.hash.TObjectIntHashMap;

import java.io.*;
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
  private static final int TRAINING_ITERS = 25;
  private static final double TRAINING_STEP_COEFF = 2e-2;
  private static final double G_DISCOUNT = 1;
  private static final FastRandom rng = new FastRandom();
  public static final int DIM = 100;
  public static final int MAX_COUNT = 100;
  public static final String WD = ".";

  private double train(Map<Generator, TIntFloatHashMap> cooccurrences, Mx leftVectors, Mx rightVectors) {
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
        final Generator generators = entry.getKey();
        final TIntFloatHashMap freqs = entry.getValue();
        final Vec totalLeft = new ArrayVec(leftVectors.columns());
        final double[] scoreLocal = new double[]{0, 0, 0};
        freqs.forEachEntry((j, X_ij) -> {
          final Vec right = rightVectors.row(j);
          final double asum = generators.stream().mapToDouble(i -> VecTools.multiply(leftVectors.row(i), right)).sum();
          final double bias = generators.stream().mapToDouble(biases::get).sum() + biases.get(j + leftVectors.rows());
          final double diff = bias + asum - Math.log(X_ij);
          final double weight = weightingFunc(X_ij);
          final double v = weight * diff * diff;
          scoreLocal[0] += v;
          scoreLocal[1] += weight;
          scoreLocal[2]++;
          if (Double.isNaN(v))
            System.out.println();
          VecTools.fill(totalLeft, 0.);
          generators.stream().forEach(i -> { // generators update
            final Vec left = leftVectors.row(i);
            VecTools.append(totalLeft, left);
            final Vec softMax = softMaxLeft.row(i);
            IntStream.range(0, left.dim()).forEach(id -> {
              final double d = weight * diff * right.get(id);
              left.adjust(id, - TRAINING_STEP_COEFF * d / Math.sqrt(softMax.get(id)));
              softMax.set(id, softMax.get(id) * G_DISCOUNT + d * d);
            });
            final double biasStep = weight * diff;// * biases.get(i);
            biases.adjust(i, -TRAINING_STEP_COEFF * biasStep / Math.sqrt(softMaxBias.get(i)));
            softMaxBias.set(i, softMaxBias.get(i) * G_DISCOUNT + MathTools.sqr(biasStep));
          });

          { // generated update
            final Vec softMax = softMaxRight.row(j);
            IntStream.range(0, right.dim()).forEach(id -> {
              final double d = weight * diff * totalLeft.get(id);
              right.adjust(id, -TRAINING_STEP_COEFF * d / Math.sqrt(softMax.get(id)));
              softMax.set(id, softMax.get(id) * G_DISCOUNT + d * d);
            });

            final double biasStep = weight * diff;// * biases.get(j);
            biases.adjust(j + leftVectors.rows(), -TRAINING_STEP_COEFF * biasStep / Math.sqrt(softMaxBias.get(j + leftVectors.rows())));
            softMaxBias.set(j + leftVectors.rows(), softMaxBias.get(j + leftVectors.rows()) * G_DISCOUNT + MathTools.sqr(biasStep));
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
    return x < MAX_COUNT ? Math.pow((x / MAX_COUNT), 0.75) : 1;
  }

  public static void main(String[] args) throws Exception {
    final Map<Generator, TIntFloatHashMap> cooccurrences = new HashMap<>();
    final List<CharSeq> tags = new ArrayList<>();
    final List<CharSeq> subs = new ArrayList<>();
    final TIntIntHashMap tagsFreq = new TIntIntHashMap();
    final TObjectIntHashMap<CharSeq> invTags = new TObjectIntHashMap<>(100000, 0.8f, -1);
    final TObjectIntHashMap<CharSeq> invSubs = new TObjectIntHashMap<>(100000, 0.8f, -1);

    try (Reader freqRd = Files.newBufferedReader(Paths.get(WD + "/tag-freqs.txt"))) {
      CsvRow.read(freqRd).forEach(row -> {
        if (tags.size() >= 300000)
          return;
        final CharSeq tag = CharSeq.intern(row.at("tag"));
        invTags.put(tag, tags.size());
        tags.add(tag);
      });
    }

    try (final InputStreamReader reader = new InputStreamReader(new GZIPInputStream(Files.newInputStream(Paths.get(WD + "/search_sessions_tags.csv.gz"))),
        StandardCharsets.UTF_8)) {
      final List<Pair<Generator, Long>> currentQuery = new ArrayList<>();
      final Holder<CharSeq> currentUser = new Holder<>();
      final long[] counter = new long[]{0};
      Interval.start();
      TDoubleArrayList times = new TDoubleArrayList();
      CsvTools.csvLines(reader, ',', '"', '\\', true).forEach(row -> {
        if (++counter[0] % 1000000 == 0) {
          times.add(Interval.time());
          times.sort();
          System.out.print("\r" + counter[0] + " lines processed for: " + Interval.time() + " median: " + times.get(times.size() / 2));
          Interval.start();
        }
        final CharSeq user = row.at("user");
        if (!user.equals(currentUser.getValue())) {
          currentQuery.clear();
          currentUser.setValue(user);
        }
        final CharSeq query = row.at("query");
        final CharSeq tag = row.at("tag");
        if (query != null) {
          currentQuery.add(Pair.create(new Generator(CharSeqTools.split(query, " ", false).map(CharSeqTools::toLowerCase).mapToInt(part -> {
            final CharSequence trim = CharSeqTools.trim(part);
            int subIdx = invSubs.get(trim);
            if (subIdx > 0)
              return subIdx;
            final CharSeq partIntern = CharSeq.intern(trim);
            invSubs.put(partIntern, subIdx = subs.size());
            subs.add(partIntern);
            return subIdx;
          }).toArray(), subs), row.asLong("ts")));
        }
        else if (tag != null) {
          final long time = row.asLong("ts");
          int tagIdx = invTags.get(tag);
          if (tagIdx < 0)
            return;
          tagsFreq.adjustOrPutValue(tagIdx, 1, 1);

          final Iterator<Pair<Generator, Long>> it = currentQuery.iterator();
          while (it.hasNext()) {
            Pair<Generator, Long> next = it.next();
            final long minutes = TimeUnit.MILLISECONDS.toMinutes(time - next.second);
            if (minutes > 30) {
              it.remove();
              continue;
            }
            final TIntFloatHashMap map = cooccurrences.computeIfAbsent(next.first, (key) -> new TIntFloatHashMap(1, 0.9f));
            map.adjustOrPutValue(tagIdx, 1.f/(1.f + minutes), 1.f/(1.f + minutes));
//            map.trimToSize();
          }
        }
      });
    }
    System.out.println();

    try(Writer coocWr = Files.newBufferedWriter(Paths.get(WD + "/cooccurrences.txt"))) {
      final List<Generator> entries = new ArrayList<>(cooccurrences.keySet());
      entries.sort(Comparator.comparingInt(e -> -cooccurrences.get(e).size()));
      for (Generator entry : entries) {
        coocWr.append('[');
        for (int i = 0; i < entry.length(); i++) {
          if (i > 0)
            coocWr.append(", ");
          coocWr.append(entry.word(i));
        }
        coocWr.append("]: ");

        final TIntFloatHashMap map = cooccurrences.get(entry);
        final int[] keys = map.keys();
        final float[] values = map.values();
        ArrayTools.parallelSort(values, keys, 0, keys.length);
        for (int i = 0; i < keys.length; i++) {
          if (i > 0)
            coocWr.append(", ");
          coocWr.append('[').append(tags.get(keys[keys.length - i - 1])).append(']').append('@').append(Float.toString(values[values.length - i - 1]));
        }
        coocWr.append('\n');
      }
    }

    final GloveLikeEmbedding embedding = new GloveLikeEmbedding();
    Mx subsVec = new VecBasedMx(subs.size(), GloveLikeEmbedding.DIM);
    Mx tagsVec = new VecBasedMx(tags.size(), GloveLikeEmbedding.DIM);

    VecTools.fillUniformPlus(subsVec, rng, 1e-3);
    VecTools.fillUniformPlus(tagsVec, rng, 1e-3);

    System.out.println("Starting optimization of " + tags.size() + " tags for " + subs.size() + " words");

    embedding.train(cooccurrences, subsVec, tagsVec);
    embedding.saveModel(tags, tagsVec, Paths.get(WD + "/tags-vec-" + DIM + ".txt"));
    embedding.saveModel(subs, subsVec, Paths.get(WD + "/subs-vec-" + DIM + ".txt"));
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

  static class Generator{
    private final int[] words;
    private final List<CharSeq> subs;

    Generator(int[] words, List<CharSeq> subs) {
      this.words = words;
      this.subs = subs;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o)
        return true;
      if (!(o instanceof Generator))
        return false;
      Generator generator = (Generator) o;
      return Arrays.equals(words, generator.words);
    }

    @Override
    public int hashCode() {
      return Arrays.hashCode(words);
    }

    public int length() {
      return words.length;
    }

    public int get(int i) {
      return words[i];
    }

    @Override
    public String toString() {
      StringBuilder builder = new StringBuilder();
      builder.append("[");
      for (int i = 0; i < words.length; i++) {
        if (i > 0)
          builder.append(',');
        builder.append(subs.get(words[i]));
      }
      builder.append(']');
      return builder.toString();
    }

    public CharSequence word(int i) {
      return subs.get(words[i]);
    }

    public IntStream stream() {
      return IntStream.of(words);
    }
  }
}
