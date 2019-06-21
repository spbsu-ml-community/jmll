package com.expleague.ml.embedding.impl;

import com.expleague.commons.func.IntDoubleConsumer;
import com.expleague.commons.seq.CharSeq;
import com.expleague.commons.seq.CharSeqTools;
import com.expleague.commons.seq.LongSeq;
import com.expleague.commons.seq.LongSeqBuilder;
import com.expleague.ml.embedding.Embedding;
import gnu.trove.list.TLongList;
import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.list.array.TLongArrayList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Reader;
import java.io.Writer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.LongFunction;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.LongStream;
import java.util.zip.GZIPOutputStream;

public abstract class CoocBasedBuilder extends EmbeddingBuilderBase {
  protected static final Logger log = LoggerFactory.getLogger(CoocBasedBuilder.class.getName());
  private static final int CAPACITY = 50_000_000;

  private List<LongSeq> cooc;
  private boolean coocReady = false;

  protected abstract Embedding<CharSeq> fit();

  protected List<CharSeq> dict() {
    return wordsList;
  }

  protected void cooc(int i, IntDoubleConsumer consumer) {
    cooc.get(i).stream().forEach(packed ->
        consumer.accept((int)(packed >>> 32), Float.intBitsToFloat((int)(packed & 0xFFFFFFFFL)))
    );
  }

  protected int index(CharSequence word) {
    return wordsIndex.get(CharSeq.create(word));
  }

  protected LongSeq cooc(int i) {
    return cooc.get(i);
  }

  protected synchronized void cooc(int i, LongSeq set) {
    if (i > cooc.size()) {
      for (int k = cooc.size(); k <= i; k++) {
        cooc.add(new LongSeq());
      }
    }
    cooc.set(i, set);
  }

  protected float unpackWeight(LongSeq cooc, int v) {
    return Float.intBitsToFloat((int) (cooc.longAt(v) & 0xFFFFFFFFL));
  }

  protected int unpackB(LongSeq cooc, int v) {
    return (int) (cooc.longAt(v) >>> 32);
  }

  protected double countWordsProbabs(TDoubleArrayList wordsProbabsLeft, TDoubleArrayList wordsProbabsRight) {
    double[] X_sum = new double[1];
    X_sum[0] = 0d;
    final int vocab_size = dict().size();
    wordsProbabsLeft.fill(0, vocab_size, 0d);
    wordsProbabsRight.fill(0, vocab_size, 0d);
    IntStream.range(0, vocab_size).forEach(i -> {
      cooc(i, (j, X_ij) -> {
        wordsProbabsLeft.set(i, wordsProbabsLeft.get(i) + X_ij);
        //wordsProbabsLeft.set(j, wordsProbabsLeft.get(j) + X_ij);
        wordsProbabsRight.set(j, wordsProbabsRight.get(j) + X_ij);
        X_sum[0] = X_sum[0] + X_ij;
      });
    });
    return X_sum[0];
  }


  @Override
  @SuppressWarnings("Duplicates")
  public Embedding<CharSeq> build() {
    try {
      log.info("==== Dictionary phase ====");
      long time = System.nanoTime();
      acquireDictionary();
      log.info("Dictionary size is " + dict().size());
      log.info("==== " + TimeUnit.NANOSECONDS.toSeconds(System.nanoTime() - time) + "s ====");
      log.info("==== Cooccurrences phase ====");
      time = System.nanoTime();
      acquireCooccurrences();
      log.info("==== " + TimeUnit.NANOSECONDS.toSeconds(System.nanoTime() - time) + "s ====");
      log.info("==== Training phase ====");
      time = System.nanoTime();
      final Embedding<CharSeq> result = fit();
      log.info("==== " + TimeUnit.NANOSECONDS.toSeconds(System.nanoTime() - time) + "s ====");
      return result;
    }
    catch (Exception e) {
      if (e instanceof RuntimeException)
        throw (RuntimeException) e;
      else
        throw new RuntimeException(e);
    }
  }

  protected void acquireCooccurrences() throws IOException {
    final Path coocPath = Paths.get(this.path.getParent().toString(), strip(this.path.getFileName()) + "." + wtype().name().toLowerCase() + "-" + wleft() + "-" + wright() + "-" + minCount() + ".cooc");
    try {
      final LongSeq[] cooc = new LongSeq[wordsList.size()];
      Reader coocReader = readExisting(coocPath);
      if (coocReader != null) {
        log.info("Reading existing cooccurrences");
        CharSeqTools.llines(coocReader, true).forEach(line -> {
          final LongSeqBuilder values = new LongSeqBuilder(wordsList.size());
          final CharSeq[] wordWeightPair = new CharSeq[3];
          CharSeqTools.split(line.line, " ", false)
              .skip(1)
              .map(part -> CharSeqTools.split(part, ':', wordWeightPair))
              .forEach(split -> values.add(((long)CharSeqTools.parseInt(split[0])) << 32 | Float.floatToIntBits(CharSeqTools.parseFloat(split[2]))));
          cooc[line.number] = values.build();
        });
        this.cooc = new ArrayList<>(Arrays.asList(cooc));
        coocReady = true;
      }
    }
    catch (IOException ioe) {
      log.warn("Unable to read : " + coocPath, ioe);
    }

    if (!coocReady) {
      log.info("Generating cooccurrences for " + this.path);
      final long startTime = System.nanoTime();
      final Lock[] rowLocks = IntStream.range(0, wordsList.size()).mapToObj(i -> new ReentrantLock()).toArray(Lock[]::new);
      final List<TLongList> accumulators = new ArrayList<>();
      cooc = IntStream.range(0, wordsList.size()).mapToObj(i -> LongSeq.empty()).collect(Collectors.toList());

      try (final LongStream stream = positionsStream()) {
        stream.parallel()/*.peek(p -> {
          System.out.println(dict().get(unpackA(p)) + "->" + dict().get(unpackB(p)) + "=" + unpackDist(p));
        })*/.mapToObj(new LongFunction<TLongList>() {
          volatile TLongList accumulator;

          @Override
          public TLongList apply(long value) {
            if (accumulator == null || accumulator.size() >= CAPACITY) {
              synchronized (this) {
                if (accumulator == null || accumulator.size() >= CAPACITY) {
                  final TLongList accumulator = this.accumulator;
                  accumulators.add(this.accumulator = new TLongArrayList(CAPACITY));
                  return accumulator;
                }
              }
            }
            accumulator.add(value);
            return null;
          }
        }).filter(Objects::nonNull).peek(accumulators::remove).peek(TLongList::sort).forEach(acc -> merge(rowLocks, (TLongArrayList) acc));
        accumulators.parallelStream().peek(TLongList::sort).forEach(acc -> merge(rowLocks, (TLongArrayList) acc));
      }
      log.info("Generated for " + TimeUnit.NANOSECONDS.toSeconds(System.nanoTime() - startTime) + "s");

      final Path coocOut = Paths.get(coocPath.toString() + ".gz");
      try (Writer coocWriter = new OutputStreamWriter(new GZIPOutputStream(Files.newOutputStream(coocOut)))) {
        log.info("Writing cooccurrences to: " + coocOut);
        for (int i = 0; i < this.cooc.size(); i++) {
          final LongSeq row = this.cooc.get(i);
          final StringBuilder builder = new StringBuilder();
          builder.append(dict().get(i)).append('\t');
          row.stream().forEach(packed -> {
            final int wordId = (int)(packed >>> 32);
            builder.append(wordId).append(':').append(dict().get(wordId)).append(':').append(CharSeqTools.ppDouble(Float.intBitsToFloat(((int)(packed & 0xFFFFFFFFL))))).append(' ');
          });
          coocWriter.append(builder, 0, builder.length() - 1).append('\n');
        }
      }
      catch (IOException ioe) {
        log.warn("Unable to write dictionary to " + coocOut, ioe);
      }
      coocReady = true;
    }
  }

  private void merge(Lock[] rowLocks, TLongArrayList acc) {
    final int size = acc.size();
    final float[] weights = new float[256];
    IntStream.range(0, 256).forEach(i -> weights[i] = (float)wtype().weight(i > 126 ? -256 + i : i));

    LongSeq prevRow = null;
    final LongSeqBuilder updatedRow = new LongSeqBuilder(wordsList.size());
    int prevA = -1;
    int pos = 0; // insertion point
    int prevLength = 0;
    try {
      for (int i = 0; i < size; i++) {
        long next = acc.getQuick(i);
        final long currentPairMasked = next & 0xFFFFFFFFFFFFFF00L;
        final int a = unpackA(next);
        final int b = unpackB(next);
        float weight = weights[unpackDist(next)];
        while (++i < size && ((next = acc.getQuick(i)) & 0xFFFFFFFFFFFFFF00L) == currentPairMasked) {
          weight += weights[unpackDist(next)];
        }
        if (i < size)
          i--;

        if (a != prevA) {
          if (prevA >= 0) {
            updatedRow.addAll(prevRow.sub(pos, prevLength));
            cooc.set(prevA, updatedRow.build(prevRow.data(), 0.2, 100));
            rowLocks[prevA].unlock();
          }
          prevA = a;
          prevRow = cooc.get(a);
          prevLength = prevRow.length();
          pos = 0;
          rowLocks[a].lock();
        }
        long prevPacked;
        while (pos < prevLength) { // merging previous version of the cooc row with current data
          prevPacked = prevRow.longAt(pos);
          int prevB = (int)(prevPacked >>> 32);
          if (prevB >= b) {
            if (prevB == b) { // second entry matches with the merged one
              weight += Float.intBitsToFloat((int) (prevPacked & 0xFFFFFFFFL));
              pos++;
            }
            break;
          }

          updatedRow.append(prevPacked);
          pos++;
        }
        final long repacked = (((long)b) << 32) | Float.floatToIntBits(weight);
        updatedRow.append(repacked);
      }
      //noinspection ConstantConditions
      updatedRow.addAll(prevRow.sub(pos, prevLength));
      cooc.set(prevA, updatedRow.build(prevRow.data(), 0.2, 100));
    }
    finally {
      rowLocks[prevA].unlock();
    }
  }
}
