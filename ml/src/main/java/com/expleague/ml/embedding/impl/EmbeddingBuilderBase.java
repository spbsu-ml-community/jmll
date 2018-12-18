package com.expleague.ml.embedding.impl;

import com.expleague.commons.csv.CsvRow;
import com.expleague.commons.csv.WritableCsvRow;
import com.expleague.commons.func.Functions;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecIterator;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.mx.SparseMx;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.math.vectors.impl.vectors.SparseVec;
import com.expleague.commons.seq.CharSeq;
import com.expleague.commons.seq.CharSeqArray;
import com.expleague.commons.seq.CharSeqComposite;
import com.expleague.commons.seq.CharSeqTools;
import com.expleague.ml.embedding.Embedding;
import gnu.trove.iterator.TLongIterator;
import gnu.trove.iterator.TObjectIntIterator;
import gnu.trove.list.TLongList;
import gnu.trove.list.array.TDoubleArrayList;
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
import java.text.BreakIterator;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.LongFunction;
import java.util.function.Supplier;
import java.util.stream.IntStream;
import java.util.stream.LongStream;
import java.util.stream.Stream;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

public abstract class EmbeddingBuilderBase implements Embedding.Builder<CharSeq> {
  private static final Logger log = LoggerFactory.getLogger(EmbeddingBuilderBase.class.getName());
  public static final int CAPACITY = 10_000_000;
  private Path path;

  private int minCount = 5;
  private int windowLeft = 15;
  private int windowRight = 15;
  private int iterations = 25;
  private double step = 0.01;

  private List<CharSeq> wordsList = new ArrayList<>();
  private TObjectIntMap<CharSeq> wordsIndex = new TObjectIntHashMap<>(50_000, 0.6f, -1);
  private boolean dictReady;
  private SparseMx cooc;
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
  public Embedding.Builder<CharSeq> leftWindow(int wnd) {
    this.windowLeft = wnd;
    return this;
  }

  @Override
  public Embedding.Builder<CharSeq> rightWindow(int wnd) {
    this.windowRight = wnd;
    return this;
  }

  @Nullable
  private Reader readExisting(Path path) throws IOException {
    if (Files.exists(path))
      return Files.newBufferedReader(path);
    else if (Files.exists(Paths.get(path.toString() + ".gz")))
      return new InputStreamReader(new GZIPInputStream(Files.newInputStream(Paths.get(path.toString() + ".gz"))), StandardCharsets.UTF_8);
    return null;
  }

  protected abstract Embedding<CharSeq> fit();

  protected List<CharSeq> dict() {
    return wordsList;
  }

  protected SparseMx cooc() {
    return cooc;
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
    cooc = new SparseMx(wordsList.size(), wordsList.size());
    final Path coocPath = Paths.get(this.path.getParent().toString(), strip(this.path.getFileName()) + ".cooc-" + windowLeft + "-" + windowRight + "-" + minCount);
    try {
      Reader coocReader = readExisting(coocPath);
      if (coocReader != null) {
        log.info("Reading existing cooccurrences");
        CharSeqTools.llines(coocReader, true).forEach(line -> {
          final TIntArrayList indices = new TIntArrayList();
          final TDoubleArrayList values = new TDoubleArrayList();
          final CharSeq[] wordWeightPair = new CharSeq[2];
          CharSeqTools.split(line.line, " ", false).forEach(part -> {
            CharSequence[] split = CharSeqTools.split(part, ':', wordWeightPair);
            indices.add(CharSeqTools.parseInt(split[0]));
            values.add(CharSeqTools.parseDouble(split[1]));
          });
          cooc.setRow(line.number, new SparseVec(wordsList.size(), indices.toArray(), values.toArray()));
        });
        coocReady = true;
      }
    }
    catch (IOException ioe) {
      log.warn("Unable to read : " + coocPath, ioe);
    }

    if (!coocReady) {
      log.info("Generating cooccurrences for " + this.path);
      final Lock[] rowLocks = IntStream.range(0, wordsList.size()).mapToObj(i -> new ReentrantLock()).toArray(Lock[]::new);

      final List<TLongList> accumulators = new ArrayList<>();
      long[] counters = new long[]{0};
      source().parallel().map(line -> {
        if (line instanceof CharSeqComposite)
          line = new CharSeqArray(line.toCharArray());
        synchronized (counters) {
          if (++counters[0] % 10000 == 0)
            log.info(counters[0] + " lines processed");
        }
        BreakIterator breakIterator = BreakIterator.getWordInstance();
        breakIterator.setText(line.it());
        int lastIndex = breakIterator.first();
        final TIntArrayList queue = new TIntArrayList(1000);

        while (BreakIterator.DONE != lastIndex) {
          int firstIndex = lastIndex;
          lastIndex = breakIterator.next();
          if (lastIndex != BreakIterator.DONE && Character.isLetterOrDigit(line.charAt(firstIndex))) {
            final CharSeq word = line.sub(firstIndex, lastIndex);
            int wordId = wordsIndex.get(word);
            if (wordId != wordsIndex.getNoEntryValue())
              queue.add(wordId);
          }
        }
        return queue;
      }).map(queue -> {
        TLongList result = new TLongArrayList(queue.size() * (windowLeft + windowRight));
        for (int i = 0; i < queue.size(); i++) {
          final int indexedId = queue.getQuick(i);
          final int rightLimit = Math.min(queue.size(), i + windowRight + 1);
          final int leftLimit = Math.max(0, i - windowLeft);
          for (int idx = leftLimit; idx < rightLimit; idx++) {
            if (idx == i)
              continue;
            result.add(pack(indexedId, queue.getQuick(idx), (byte) Math.abs(i - idx)));
          }
        }
        return result;
      }).flatMapToLong(entries -> LongStream.of(entries.toArray())).mapToObj(new LongFunction<TLongList>() {
        volatile TLongList accumulator;
        @Override
        public TLongList apply(long value) {
          if (accumulator == null || accumulator.size() >= CAPACITY) {
            synchronized (this) {
              final TLongList accumulator = this.accumulator;
              accumulators.add(this.accumulator = new TLongArrayList(CAPACITY));
              return accumulator;
            }
          }
          accumulator.add(value);
          return null;
        }
      }).filter(Objects::nonNull).peek(accumulators::remove).peek(TLongList::sort).forEach(acc -> merge(rowLocks, acc));
      accumulators.parallelStream().forEach(acc -> merge(rowLocks, acc));

      final Path coocOut = Paths.get(coocPath.toString() + ".gz");
      try (Writer coocWriter = new OutputStreamWriter(new GZIPOutputStream(Files.newOutputStream(coocOut)))) {
        log.info("Writing cooccurrences to: " + coocOut);
        for (int i = 0; i < cooc.rows(); i++) {
          final Vec row = cooc.getRow(i);
          if (row != null) {
            final VecIterator nz = row.nonZeroes();
            boolean started = false;
            while (nz.advance()) {
              if (started)
                coocWriter.append(' ');
              else
                started = true;
              coocWriter.append(Integer.toString(nz.index()))
                  .append(':')
                  .append(CharSeqTools.ppDouble(nz.value()));
            }
          }
          coocWriter.append('\n');
        }
      }
      catch (IOException ioe) {
        log.warn("Unable to write dictionary to " + coocOut, ioe);
      }
    }
  }

  private void merge(Lock[] rowLocks, TLongList acc) {
    final TLongIterator iterator = acc.iterator();
    int prev = -1;
    boolean last = false;
    SparseVec row = new SparseVec(wordsList.size());
    while (iterator.hasNext() || last) {
      final long next = !last ? iterator.next() : -1;
      final int a = unpackA(next);
      if (a != prev || last) {
        if (prev >= 0 && row.size() > 0) {
          rowLocks[prev].lock();
          final Vec updateRow = cooc.row(prev);
          VecTools.append(updateRow, row);
          if (updateRow instanceof SparseVec) {
            if (((SparseVec) updateRow).size() / (double)updateRow.dim() > 0.3) {
              Vec dense = new ArrayVec(updateRow.dim());
              VecTools.assign(dense, updateRow);
//              log.info("Converting word " + wordsList.get(prev) + " to dense");
              cooc.setRow(prev, dense);
            }
          }
          rowLocks[prev].unlock();
          VecTools.scale(row, 0.);
        }
        prev = a;
      }
      row.adjust(unpackB(next), 1. / (unpackFreq(next)));
      last = !last && !iterator.hasNext();
    }
  }

  private int unpackA(long next) {
    return (int)(next >>> 36);
  }

  private int unpackB(long next) {
    return ((int)(next >>> 8)) & 0x0FFFFFFF;
  }

  private int unpackFreq(long next) {
    return (int)(0xFF & next);
  }

  private long pack(long a, long b, byte dist) {
    return (a << 36) | (b << 8) | ((long)dist);
  }

  private void acquireDictionary() throws IOException {
    final Path dictPath = Paths.get(this.path.getParent().toString(), strip(this.path.getFileName()) + ".dict");
    try {
      Reader dictReader = readExisting(dictPath);
      if (dictReader != null) {
        log.info("Reading existing dictionary");
        try (Stream<CsvRow> dictStream = CsvRow.read(dictReader)) {
          dictStream.filter(row -> row.asInt("freq") > minCount).map(row -> CharSeq.intern(row.at("word"))).forEach(word -> {
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
      source().forEach(line -> {
        BreakIterator breakIterator = BreakIterator.getWordInstance();
        breakIterator.setText(line.it());
        int lastIndex = breakIterator.first();
        while (BreakIterator.DONE != lastIndex) {
          int firstIndex = lastIndex;
          lastIndex = breakIterator.next();
          if (lastIndex != BreakIterator.DONE && Character.isLetterOrDigit(line.charAt(firstIndex))) {
            final CharSeq word = CharSeq.create(CharSeqTools.toLowerCase(line.sub(firstIndex, lastIndex)));
            if (word.length() != 1 && word.stream().anyMatch(Character::isLetter)) {
              wordsCount.adjustOrPutValue(word, 1, 1);
            }
          }
        }
      });

      final Path dictOut = Paths.get(dictPath.toString() + ".gz");
      final Supplier<WritableCsvRow> factory = CsvRow.factory("word", "freq");
      try (Writer dictWriter = new OutputStreamWriter(new GZIPOutputStream(Files.newOutputStream(dictOut)))) {
        log.info("Writing dictionary to: " + dictOut);
        dictWriter.append(factory.get().names().toString()).append('\n');
        dictWriter.append(Integer.toString(wordsList.size()));
        wordsCount.forEachEntry((word, freq) -> {
          factory.get().set("word", word).set("freq", freq).writeln(dictWriter);
          return true;
        });
      }
      catch (IOException ioe) {
        log.warn("Unable to write dictionary to " + dictOut, ioe);
      }

      for (TObjectIntIterator<CharSeq> it = wordsCount.iterator(); it.hasNext();) {
        it.advance();
        if (it.value() >= minCount) {
          wordsIndex.put(it.key(), wordsList.size());
          wordsList.add(it.key());
        }
      }
      coocReady = true;
    }
  }

  protected Stream<CharSeq> source() throws IOException {
    if (path.getFileName().toString().endsWith(".gz"))
      return CharSeqTools.lines(new InputStreamReader(new GZIPInputStream(Files.newInputStream(Paths.get(path.toString() + ".gz"))), StandardCharsets.UTF_8));
    return CharSeqTools.lines(Files.newBufferedReader(path));
  }

  protected String strip(Path fileName) {
    final String name = fileName.toString();
    if (name.endsWith(".gz"))
      return name.substring(0, name.length() - ".gz".length());
    return name;
  }
}
