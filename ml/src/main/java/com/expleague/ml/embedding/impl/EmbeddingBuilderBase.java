package com.expleague.ml.embedding.impl;

import com.expleague.commons.csv.CsvRow;
import com.expleague.commons.csv.WritableCsvRow;
import com.expleague.commons.func.IntDoubleConsumer;
import com.expleague.commons.seq.CharSeq;
import com.expleague.commons.seq.CharSeqTools;
import com.expleague.commons.seq.LongSeq;
import com.expleague.commons.seq.LongSeqBuilder;
import com.expleague.ml.embedding.Embedding;
import gnu.trove.list.TLongList;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.list.array.TLongArrayList;
import gnu.trove.map.TObjectIntMap;
import gnu.trove.map.hash.TObjectIntHashMap;
import org.jetbrains.annotations.Nullable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Consumer;
import java.util.function.IntFunction;
import java.util.function.LongFunction;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.LongStream;
import java.util.stream.Stream;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

public abstract class EmbeddingBuilderBase implements Embedding.Builder<CharSeq> {
  private static final Logger log = LoggerFactory.getLogger(EmbeddingBuilderBase.class.getName());
  public static final int CAPACITY = 50_000_000;
  private Path path;

  private int minCount = 5;
  private int windowLeft = 15;
  private int windowRight = 15;
  private Embedding.WindowType windowType = Embedding.WindowType.LINEAR;
  private int iterations = 25;
  private double step = 0.01;

  private List<CharSeq> wordsList = new ArrayList<>();
  private TObjectIntMap<CharSeq> wordsIndex = new TObjectIntHashMap<>(50_000, 0.6f, -1);
  private boolean dictReady;
  private List<LongSeq> cooc;
  private boolean coocReady = false;

  @Override
  public Embedding.Builder<CharSeq> file(Path path) {
    this.path = path;
    return this;
  }

  @Override
  public Embedding.Builder<CharSeq> minWordCount(int count) {
    this.minCount = count;
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

  protected int T() {
    return iterations;
  }

  protected double step() {
    return step;
  }

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

  private void acquireCooccurrences() throws IOException {
    final Path coocPath = Paths.get(this.path.getParent().toString(), strip(this.path.getFileName()) + "." + windowType.name().toLowerCase() + "-" + windowLeft + "-" + windowRight + "-" + minCount + ".cooc");
    try {
      final LongSeq[] cooc = new LongSeq[wordsList.size()];
      Reader coocReader = readExisting(coocPath);
      if (coocReader != null) {
        log.info("Reading existing cooccurrences");
        CharSeqTools.llines(coocReader, true).forEach(line -> {
          final LongSeqBuilder values = new LongSeqBuilder(wordsList.size());
          final CharSeq[] wordWeightPair = new CharSeq[2];
          CharSeqTools.split(line.line, " ", false)
              .map(part -> CharSeqTools.split(part, ':', wordWeightPair))
              .forEach(split -> values.add(((long)CharSeqTools.parseInt(split[0])) << 32 | Float.floatToIntBits(CharSeqTools.parseFloat(split[1]))));
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
      final CharSeq newLine = CharSeq.create("777newline777");
      wordsIndex.put(newLine, Integer.MAX_VALUE);
      source().peek(new Consumer<CharSeq>() {
        long line = 0;
        long time = System.nanoTime();
        @Override
        public synchronized void accept(CharSeq l) {
          if ((++line) % 10000 == 0) {
            log.info(line + " lines processed for " + TimeUnit.NANOSECONDS.toSeconds(System.nanoTime() - time) + "s");
            time = System.nanoTime();
          }
        }
      }).map(line -> (CharSeq)CharSeqTools.concat(line, " ", newLine)).flatMap(CharSeqTools::words).map(this::normalize).mapToInt(wordsIndex::get).filter(idx -> idx >= 0).mapToObj(new IntFunction<LongStream>() {
        final TIntArrayList queue = new TIntArrayList(1000);
        final Lock lock = new ReentrantLock();
        int offset = 0;
        @Override
        public LongStream apply(int idx) {
          if (idx == Integer.MAX_VALUE) { // new line
            queue.resetQuick();
            offset = 0;
            return LongStream.empty();
          }
          lock.lock();
          int pos = queue.size();
          final long[] out = new long[windowLeft + windowRight];
          int outIndex = 0;
          for (int i = offset; i < pos; i++) {
            byte distance = (byte)(pos - i);
            if (distance <= windowRight)
              out[outIndex++] = pack(queue.getQuick(i), idx, distance);
            if (distance <= windowLeft)
              out[outIndex++] = pack(idx, queue.getQuick(i), (byte)-distance);
          }
          queue.add(idx);
          if (queue.size() > Math.max(windowLeft, windowRight)) {
            offset++;
            if (offset > 1000 - Math.max(windowLeft, windowRight)) {
              queue.remove(0, offset);
              offset = 0;
            }
          }
          lock.unlock();
          return Arrays.stream(out, 0, outIndex);
        }
      }).flatMapToLong(entries -> entries).parallel()/*.peek(p -> {
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
      }).filter(Objects::nonNull).peek(accumulators::remove).peek(TLongList::sort).forEach(acc -> merge(rowLocks, (TLongArrayList)acc));
      accumulators.parallelStream().peek(TLongList::sort).forEach(acc -> merge(rowLocks, (TLongArrayList)acc));
      log.info("Generated for " + TimeUnit.NANOSECONDS.toSeconds(System.nanoTime() - startTime) + "s");
      wordsIndex.remove(newLine);

      final Path coocOut = Paths.get(coocPath.toString() + ".gz");
      try (Writer coocWriter = new OutputStreamWriter(new GZIPOutputStream(Files.newOutputStream(coocOut)))) {
        log.info("Writing cooccurrences to: " + coocOut);
        for (int i = 0; i < this.cooc.size(); i++) {
          final LongSeq row = this.cooc.get(i);
          final StringBuilder builder = new StringBuilder();
          row.stream().forEach(packed ->
              builder.append(packed >>> 32).append(':').append(CharSeqTools.ppDouble(Float.intBitsToFloat(((int) (packed & 0xFFFFFFFFL))))).append(' ')
          );
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
    IntStream.range(0, 256).forEach(i -> weights[i] = (float)windowType.weight(i > 126 ? -256 + i : i));

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
        final long limit = (long) b << 32;
        while (pos < prevLength) { // merging previous version of the cooc row with current data
          prevPacked = prevRow.longAt(pos);
          if (prevPacked >= limit) {
            if (prevPacked == limit) { // second entry matches with the merged one
              weight += Float.intBitsToFloat((int) (prevPacked & 0xFFFFFFFFL));
              pos++;
            }
            break;
          }

          updatedRow.append(prevPacked);
          pos++;
        }
        final long repacked = limit | Float.floatToIntBits(weight);
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

  private int unpackA(long next) {
    return (int)(next >>> 36);
  }

  private int unpackB(long next) {
    return ((int)(next >>> 8)) & 0x0FFFFFFF;
  }

  private int unpackDist(long next) {
    return (int)(0xFF & next);
  }

  private long pack(long a, long b, byte dist) {
    return (a << 36) | (b << 8) | ((long) dist & 0xFF);
  }

  private void acquireDictionary() throws IOException {
    final Path dictPath = Paths.get(this.path.getParent().toString(), strip(this.path.getFileName()) + ".dict");
    try {
      Reader dictReader = readExisting(dictPath);
      if (dictReader != null) {
        log.info("Reading existing dictionary");
        try (Stream<CsvRow> dictStream = CsvRow.read(dictReader)) {
          dictStream.filter(row -> row.asInt("freq") >= minCount).map(row -> CharSeq.intern(row.at("word"))).forEach(word -> {
            wordsIndex.put(word, wordsList.size());
            wordsList.add(word);
          });
        }
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

      words.forEach(word -> {
        if (wordsCount.get(word) >= minCount) {
          wordsIndex.put(word, wordsList.size());
          wordsList.add(word);
        }
      });
      dictReady = true;
    }
  }

  private CharSeq normalize(CharSeq word) {
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

  protected float unpackWeight(LongSeq cooc, int v) {
    return Float.intBitsToFloat((int) (cooc.longAt(v) & 0xFFFFFFFFL));
  }

  protected int unpackB(LongSeq cooc, int v) {
    return (int) (cooc.longAt(v) >>> 32);
  }

  @Nullable
  private Reader readExisting(Path path) throws IOException {
    if (Files.exists(path))
      return Files.newBufferedReader(path);
    else if (Files.exists(Paths.get(path.toString() + ".gz")))
      return new InputStreamReader(new GZIPInputStream(Files.newInputStream(Paths.get(path.toString() + ".gz"))), StandardCharsets.UTF_8);
    return null;
  }

  protected class ScoreCalculator {
    private final double[] scores;
    private final double[] weights;
    private final long[] counts;

    public ScoreCalculator(int dim) {
      counts = new long[dim];
      scores = new double[dim];
      weights = new double[dim];
    }

    public void adjust(int i, int j, double weight, double value) {
      weights[i] += weight;
      scores[i] += value;
      counts[i] ++;
    }

    public double gloveScore() {
      return Arrays.stream(scores).sum() / Arrays.stream(counts).sum();
    }

    public long count() {
      return Arrays.stream(counts).sum();
    }
  }
}
