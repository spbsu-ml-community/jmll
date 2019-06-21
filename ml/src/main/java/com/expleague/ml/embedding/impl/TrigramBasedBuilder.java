package com.expleague.ml.embedding.impl;

import com.expleague.commons.csv.CsvRow;
import com.expleague.commons.csv.WritableCsvRow;
import com.expleague.commons.func.IntDoubleConsumer;
import com.expleague.commons.seq.CharSeq;
import com.expleague.commons.seq.CharSeqTools;
import com.expleague.commons.seq.LongSeq;
import com.expleague.commons.seq.LongSeqBuilder;
import com.expleague.ml.embedding.Embedding;
import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.map.TObjectDoubleMap;
import gnu.trove.map.TObjectIntMap;
import gnu.trove.map.hash.TObjectIntHashMap;

import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Reader;
import java.io.Writer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;
import java.util.zip.GZIPOutputStream;

public abstract class TrigramBasedBuilder extends CoocBasedBuilder {
  protected List<CharSeq> trigsList = new ArrayList<>();
  protected TObjectIntMap<CharSeq> trigsIndex = new TObjectIntHashMap<>(20_000, 0.6f, -1);
  private boolean trigsDictReady;

  private List<LongSeq> trigramCooc;
  private boolean trigramCoocReady = false;

  protected abstract Embedding<CharSeq> fit();

  @Override
  public Embedding<CharSeq> build() {
    try {
      log.info("==== Dictionary phase ====");
      long time = System.nanoTime();
      acquireDictionary();
      log.info("==== " + TimeUnit.NANOSECONDS.toSeconds(System.nanoTime() - time) + "s ====");
      log.info("==== Cooccurrences phase ====");
      time = System.nanoTime();
      acquireCooccurrences();
      log.info("==== " + TimeUnit.NANOSECONDS.toSeconds(System.nanoTime() - time) + "s ====");

      log.info("==== Trigrams dictionary phase ====");
      acquireTrigramDictionary();
      time = System.nanoTime();
      log.info("==== " + TimeUnit.NANOSECONDS.toSeconds(System.nanoTime() - time) + "s ====");

      log.info("==== Trigrams cooccurrences phase ====");
      acquireTrigramCooccurrences();
      time = System.nanoTime();
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

  protected void trigrCooc(int i, IntDoubleConsumer consumer) {
    trigramCooc.get(i).stream().forEach(packed ->
        consumer.accept((int)(packed >>> 32), Float.intBitsToFloat((int)(packed & 0xFFFFFFFFL)))
    );
  }

  protected List<CharSeq> trigDict() {
    return trigsList;
  }

  protected int trigrToId(CharSeq trigr) { return trigsIndex.get(trigr); }

  protected double getTrigramCooc(int i, int j) {
    final LongSeq row = trigramCooc.get(i);
    for (long packed : row.data()) {
      int unpackedJ = (int)(packed >>> 32);
      if (unpackedJ == j) {
        return (double) Float.intBitsToFloat((int) (packed & 0xFFFFFFFFL));
      }
    }
    log.warn("No such cooccurrence: " + i + " & " + j);
    return 0d;
  }

  protected double countTrigramProbabs(TDoubleArrayList trigramProbabs) {
    double[] sum = new double[1];
    sum[0] = 0d;
    final int vocab_size = trigDict().size();
    trigramProbabs.fill(0, vocab_size, 0d);
    IntStream.range(0, vocab_size).forEach(i -> {
      trigrCooc(i, (j, tr_ij) -> {
        trigramProbabs.set(i, trigramProbabs.get(i) + tr_ij);
        sum[0] = sum[0] + tr_ij;
      });
    });
    return sum[0];
  }

  private void acquireTrigramDictionary() {

    final Path trigsPath = Paths.get(this.path.getParent().toString(), strip(this.path.getFileName()) + ".trigsDict");
    try {
      Reader trigsReader = readExisting(trigsPath);
      if (trigsReader != null) {
        log.info("Reading existing trigrams dictionary");
        try (Stream<CsvRow> dictStream = CsvRow.read(trigsReader)) {
          dictStream.forEach(row -> {
            CharSeq word = CharSeq.intern(row.at("trig"));
            trigsIndex.put(word, trigsList.size());
            trigsList.add(word);
          });
        }
        trigsDictReady = true;
      }
    }
    catch (IOException ioe) {
      log.warn("Unable to read trigrams dictionary: " + trigsPath, ioe);
    }
    if (!trigsDictReady) {
      log.info("Generating trigrams dictionary for " + this.path);
      TObjectIntMap<CharSeq> wordsCount = new TObjectIntHashMap<>();
      for (int i = 0; i < dict().size(); i++) {
        final CharSeq word = dict().get(i);
        for (int k = 0; k < word.length() - 2; k++) {
          wordsCount.adjustOrPutValue(word.sub(k, k + 3), 1, 1);
        }
      }

      final Path dictOut = Paths.get(trigsPath.toString() + ".gz");
      final Supplier<WritableCsvRow> factory = CsvRow.factory("trig", "freq");
      final List<CharSeq> trigrams = new ArrayList<>(wordsCount.keySet());
      trigrams.sort(Comparator.comparingInt(wordsCount::get).reversed());
      try (Writer dictWriter = new OutputStreamWriter(new GZIPOutputStream(Files.newOutputStream(dictOut)))) {
        log.info("Writing trigrams dictionary to: " + dictOut);
        dictWriter.append(factory.get().names().toString()).append('\n');
        trigrams.forEach(trig ->
            factory.get().set("trig", trig).set("freq", wordsCount.get(trig)).writeln(dictWriter)
        );
      }
      catch (IOException ioe) {
        log.warn("Unable to write trigrams dictionary to " + dictOut, ioe);
      }
      trigrams.forEach(trig -> {
        trigsIndex.put(trig, trigsList.size());
        trigsList.add(trig);
      });
      trigsDictReady = true;
    }
  }

  protected void acquireTrigramCooccurrences() throws IOException {
    final Path trigramCoocPath = Paths.get(this.path.getParent().toString(), strip(this.path.getFileName()) + "." + wtype().name().toLowerCase() + "-" + wleft() + "-" + wright() + "-" + minCount() + ".trigsCooc");
    try {
      final LongSeq[] cooc = new LongSeq[trigsList.size()];
      Reader coocReader = readExisting(trigramCoocPath);
      if (coocReader != null) {
        log.info("Reading existing trigram cooccurrences");
        CharSeqTools.llines(coocReader, true).forEach(line -> {
          final LongSeqBuilder values = new LongSeqBuilder(trigsList.size());
          final CharSeq[] wordWeightPair = new CharSeq[3];
          CharSeqTools.split(line.line, " ", false).skip(1).map(part -> CharSeqTools.split(part, ':', wordWeightPair)).forEach(split -> values.add(((long) CharSeqTools.parseInt(split[0])) << 32 | Float.floatToIntBits(CharSeqTools.parseFloat(split[2]))));
          cooc[line.number] = values.build();
        });
        this.trigramCooc = new ArrayList<>(Arrays.asList(cooc));
        trigramCoocReady = true;
      }
    }
    catch (IOException ioe) {
      log.warn("Unable to read : " + trigramCoocPath, ioe);
    }

    if (!trigramCoocReady) {
      log.info("Generating trigram cooccurrences for " + this.path);
      final long startTime = System.nanoTime();
      trigramCooc = IntStream.range(0, trigsList.size()).mapToObj(i -> LongSeq.empty()).collect(Collectors.toList());
      trigramCooc = Collections.synchronizedList(trigramCooc);

      IntStream.range(0, dict().size()).parallel().forEach(i -> {
        CharSeq word_i = dict().get(i);
        cooc(i, (j, X_ij) -> {
          for (int k = 0; k < word_i.length() - 2; k ++) {
            final int trig_id = trigsIndex.get(word_i.subSequence(k, k + 3));
            final LongSeq prevRow = trigramCooc.get(trig_id);
            final LongSeqBuilder updatedRow = new LongSeqBuilder(wordsList.size());
            if (prevRow.length() == 0) {
              final long repacked = (((long)j) << 32) | Float.floatToIntBits((float) X_ij);
              updatedRow.append(repacked);
            } else {
              final boolean[] foundJ = {false};
              prevRow.stream().forEach(prevPacked -> {
                int prevJ = (int)(prevPacked >>> 32);
                if (prevJ < j) {
                  updatedRow.append(prevPacked);
                } else if (prevJ == j) {
                  final float weight = Float.intBitsToFloat((int) (prevPacked & 0xFFFFFFFFL)) + (float) X_ij;
                  final long repacked = (((long)j) << 32) | Float.floatToIntBits(weight);
                  updatedRow.append(repacked);
                  foundJ[0] = true;
                } else if (!foundJ[0]) {
                  final long repacked = (((long)j) << 32) | Float.floatToIntBits((float) X_ij);
                  updatedRow.append(repacked);
                  updatedRow.append(prevPacked);
                  foundJ[0] = true;
                } else {
                  updatedRow.append(prevPacked);
                }
              });
              if (!foundJ[0]) {
                final long repacked = (((long)j) << 32) | Float.floatToIntBits((float) X_ij);
                updatedRow.append(repacked);
                foundJ[0] = true;
              }
            }
            trigramCooc.set(trig_id, updatedRow.build(prevRow.data(), 0.2, 100));
          }
        });
        if (i % 10 == 0)
          System.out.println(word_i);
      });

      log.info("Generated for " + TimeUnit.NANOSECONDS.toSeconds(System.nanoTime() - startTime) + "s");
      final Path coocOut = Paths.get(trigramCoocPath.toString() + ".gz");
      try (Writer coocWriter = new OutputStreamWriter(new GZIPOutputStream(Files.newOutputStream(coocOut)))) {
        log.info("Writing trigram cooccurrences to: " + coocOut);
        for (int i = 0; i < this.trigramCooc.size(); i++) {
          final LongSeq row = this.trigramCooc.get(i);
          final StringBuilder builder = new StringBuilder();
          builder.append(trigsList.get(i)).append('\t');
          row.stream().forEach(packed -> {
            final int wordId = (int)(packed >>> 32);
            builder.append(wordId).append(':').append(dict().get(wordId)).append(':').append(CharSeqTools.ppDouble(Float.intBitsToFloat(((int)(packed & 0xFFFFFFFFL))))).append(' ');
          });
          coocWriter.append(builder, 0, builder.length() - 1).append('\n');
        }
      }
      catch (IOException ioe) {
        log.warn("Unable to write trigrams dictionary to " + coocOut, ioe);
      }
      trigramCoocReady = true;
    }
  }
}
