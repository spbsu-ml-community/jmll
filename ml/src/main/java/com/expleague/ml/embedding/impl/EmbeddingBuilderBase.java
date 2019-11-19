package com.expleague.ml.embedding.impl;

import com.expleague.commons.csv.CsvRow;
import com.expleague.commons.csv.WritableCsvRow;
import com.expleague.commons.seq.CharSeq;
import com.expleague.commons.seq.CharSeqTools;
import com.expleague.ml.embedding.Embedding;
import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.map.TObjectIntMap;
import gnu.trove.map.hash.TObjectIntHashMap;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;
import java.util.function.IntFunction;
import java.util.function.Supplier;
import java.util.stream.IntStream;
import java.util.stream.LongStream;
import java.util.stream.Stream;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

@SuppressWarnings("WeakerAccess")
public abstract class EmbeddingBuilderBase implements Embedding.Builder<CharSeq> {
  protected static final Logger log = LoggerFactory.getLogger(EmbeddingBuilderBase.class.getName());
  protected Path path;

  private int minCount = 5;
  private int windowLeft = 15;
  private int windowRight = 15;
  private Embedding.WindowType windowType = Embedding.WindowType.LINEAR;
  private int iterations = 25;
  private double step = 0.01;

  protected List<CharSeq> wordsList = new ArrayList<>();
  protected TDoubleArrayList wordsPrior = new TDoubleArrayList();
  protected TObjectIntMap<CharSeq> wordsIndex = new TObjectIntHashMap<>(50_000, 0.6f, -1);
  private boolean dictReady;
  private int progressLines = 1000;

  @Override
  public Embedding.Builder<CharSeq> file(Path path) {
    this.path = path;
    return this;
  }

  @Override
  public Embedding.Builder<CharSeq> window(Embedding.WindowType type, int left, int right) {
    this.windowLeft = left;
    this.windowRight = right;
    this.windowType = type;
    return this;
  }

  @Override
  public Embedding.Builder<CharSeq> step(double step) {
    this.step = step;
    return this;
  }

  @Override
  public Embedding.Builder<CharSeq> iterations(int count) {
    iterations = count;
    return this;
  }

  @Override
  public Embedding.Builder<CharSeq> minWordCount(int count) {
    this.minCount = count;
    return this;
  }

  @SuppressWarnings("unused")
  public void progressLines(int lines) {
    this.progressLines = lines;
  }

  protected abstract Embedding<CharSeq> fit();

  protected List<CharSeq> dict() {
    return wordsList;
  }

  protected int index(CharSequence word) {
    return wordsIndex.get(CharSeq.create(word));
  }

  protected int T() {
    return iterations;
  }

  protected double step() {
    return step;
  }

  protected int minCount() {
    return minCount;
  }

  protected Embedding.WindowType wtype() {
    return windowType;
  }

  protected int wleft() {
    return windowLeft;
  }

  protected int wright() {
    return windowRight;
  }

  protected double p(int idx) {
    return wordsPrior.get(idx);
  }

  @SuppressWarnings("Duplicates")
  @Override
  public Embedding<CharSeq> build() {
    try {
      log.info("==== Dictionary phase ====");
      long time = System.nanoTime();
      acquireDictionary();
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

  @NotNull
  protected LongStream positionsStream() throws IOException {
    LongStream coocStream;
    final CharSeq newLine = CharSeq.create("777newline777");
    wordsIndex.put(newLine, Integer.MAX_VALUE);
    coocStream = source().peek(new Consumer<CharSeq>() {
      long line = 0;
      long time = System.nanoTime();

      @Override
      public synchronized void accept(CharSeq l) {
        if ((++line) % progressLines == 0) {
          CoocBasedBuilder.log.info(line + " lines processed for " + TimeUnit.NANOSECONDS.toSeconds(System.nanoTime() - time) + "s");
          time = System.nanoTime();
        }
      }
    }).map(line -> (CharSeq) CharSeqTools.concat(line, " ", newLine)).flatMap(CharSeqTools::words).map(this::normalize).mapToInt(wordsIndex::get).filter(idx -> idx >= 0).mapToObj(new IntFunction<LongStream>() {
      final TIntArrayList queue = new TIntArrayList(1000);
      int offset = 0;

      @Override
      public synchronized LongStream apply(int idx) {
        if (idx == Integer.MAX_VALUE) { // new line
          queue.resetQuick();
          offset = 0;
          return LongStream.empty();
        }
        int pos = queue.size();
        final long[] out = new long[windowLeft + windowRight];
        int outIndex = 0;
        for (int i = offset; i < pos; i++) {
          byte distance = (byte) (pos - i);
          if (distance == 0) {
            CoocBasedBuilder.log.warn("Zero distance occured! pos: " + pos + " i: " + i);
            System.err.println("Zero distance occured! pos: " + pos + " i: " + i);
          }
          if (distance <= windowRight)
            out[outIndex++] = pack(queue.getQuick(i), idx, distance);
          if (distance <= windowLeft)
            out[outIndex++] = pack(idx, queue.getQuick(i), (byte) -distance);
        }
        queue.add(idx);
        if (queue.size() > Math.max(windowLeft, windowRight)) {
          offset++;
          if (offset > 1000 - Math.max(windowLeft, windowRight)) {
            queue.remove(0, offset);
            offset = 0;
          }
        }
        return Arrays.stream(out, 0, outIndex);
      }
    }).flatMapToLong(entries -> entries).onClose(() -> wordsIndex.remove(newLine));
    return coocStream;
  }

  @NotNull
  protected IntStream wordsIndexesStream() throws IOException {
    IntStream idxStream;
    final CharSeq newLine = CharSeq.create("777newline777");
    wordsIndex.put(newLine, Integer.MAX_VALUE);
    idxStream = source().peek(new Consumer<CharSeq>() {
      long line = 0;
      long time = System.nanoTime();

      @Override
      public synchronized void accept(CharSeq l) {
        if ((++line) % progressLines == 0) {
          CoocBasedBuilder.log.info(line + " lines processed for " + TimeUnit.NANOSECONDS.toSeconds(System.nanoTime() - time) + "s");
          time = System.nanoTime();
        }
      }
    }).map(line -> (CharSeq) CharSeqTools.concat(line, " ", newLine)).flatMap(CharSeqTools::words).map(this::normalize).mapToInt(wordsIndex::get).filter(idx -> idx >= 0)
        .onClose(() -> wordsIndex.remove(newLine));
    return idxStream;
  }

  protected int unpackA(long next) {
    return (int)(next >>> 36);
  }

  protected int unpackB(long next) {
    return ((int)(next >>> 8)) & 0x0FFFFFFF;
  }

  protected double unpackWeight(long next) {
    int dist = (int) (0xFF & next);
    return wtype().weight(dist > 126 ? -256 + dist : dist);
  }

  protected int unpackDist(long next) {
    return (int)(0xFF & next);
  }

  protected long pack(long a, long b, byte dist) {
    return (a << 36) | (b << 8) | ((long) dist & 0xFF);
  }

  @SuppressWarnings("WeakerAccess")
  protected void acquireDictionary() throws IOException {
    final Path dictPath = Paths.get(this.path.getParent().toString(), strip(this.path.getFileName()) + ".dict");
    try {
      Reader dictReader = readExisting(dictPath);
      if (dictReader != null) {
        log.info("Reading existing dictionary");
        double[] total = new double[]{0};
        try (Stream<CsvRow> dictStream = CsvRow.read(dictReader)) {
          dictStream.peek(row -> total[0] += row.asInt("freq"))
              .filter(row -> row.asInt("freq") >= minCount)
              .forEach(row -> {
                CharSeq word = CharSeq.intern(row.at("word"));
                wordsPrior.add(row.asInt("freq"));
                wordsIndex.put(word, wordsList.size());
                wordsList.add(word);
              });
        }
        wordsPrior.transformValues(v -> v / total[0]);
        dictReady = true;
      }
    }
    catch (IOException ioe) {
      log.warn("Unable to read dictionary: " + dictPath, ioe);
    }

    if (!dictReady) {
      log.info("Generating dictionary for " + this.path);
      TObjectIntMap<CharSeq> wordsCount = new TObjectIntHashMap<>();
      source().flatMap(CharSeqTools::words).filter(word -> word.stream().anyMatch(Character::isLetter)).map(this::normalize).forEach(w ->
          wordsCount.adjustOrPutValue(w, 1, 1)
      );

      final Path dictOut = Paths.get(dictPath.toString() + ".gz");
      final Supplier<WritableCsvRow> factory = CsvRow.factory("word", "freq");
      final List<CharSeq> words = new ArrayList<>(wordsCount.keySet());
      words.sort(Comparator.comparingInt(wordsCount::get).reversed());
      try (Writer dictWriter = new OutputStreamWriter(new GZIPOutputStream(Files.newOutputStream(dictOut)))) {
        log.info("Writing dictionary to: " + dictOut);
        dictWriter.append(factory.get().names().toString()).append('\n');
        words.forEach(word ->
          factory.get().set("word", word).set("freq", wordsCount.get(word)).writeln(dictWriter)
        );
      }
      catch (IOException ioe) {
        log.warn("Unable to write dictionary to " + dictOut, ioe);
      }
      final double total = IntStream.of(wordsCount.values()).mapToDouble(x -> x).sum();
      words.forEach(word -> {
        if (wordsCount.get(word) >= minCount) {
          wordsPrior.add(wordsCount.get(word) / total);
          wordsIndex.put(word, wordsList.size());
          wordsList.add(word);
        }
      });
      dictReady = true;
    }
  }

  protected CharSeq normalize(CharSeq word) {
    final int initialLength = word.length();
    int len = initialLength;
    int st = 0;

    while ((st < len) && !Character.isLetterOrDigit(word.charAt(st))) {
      st++;
    }
    while ((st < len) && !Character.isLetterOrDigit(word.charAt(len - 1))) {
      len--;
    }
    word = ((st > 0) || (len < initialLength)) ? word.subSequence(st, len) : word;

    return (CharSeq)CharSeqTools.toLowerCase(word);
  }

  protected Stream<CharSeq> source() throws IOException {
    if (path.getFileName().toString().endsWith(".gz"))
      return CharSeqTools.lines(new InputStreamReader(new GZIPInputStream(Files.newInputStream(path)), StandardCharsets.UTF_8));
    return CharSeqTools.lines(Files.newBufferedReader(path));
  }

  protected String strip(Path fileName) {
    final String name = fileName.toString();
    if (name.endsWith(".gz"))
      return name.substring(0, name.length() - ".gz".length());
    return name;
  }

  @SuppressWarnings("WeakerAccess")
  @Nullable
  protected Reader readExisting(Path path) throws IOException {
    if (Files.exists(path))
      return Files.newBufferedReader(path);
    else if (Files.exists(Paths.get(path.toString() + ".gz")))
      return new InputStreamReader(new GZIPInputStream(Files.newInputStream(Paths.get(path.toString() + ".gz"))), StandardCharsets.UTF_8);
    return null;
  }
}
