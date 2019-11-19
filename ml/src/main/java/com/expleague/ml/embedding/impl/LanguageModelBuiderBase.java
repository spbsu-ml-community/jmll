package com.expleague.ml.embedding.impl;

import com.expleague.commons.seq.*;
import com.expleague.ml.embedding.Embedding;
import gnu.trove.list.array.TIntArrayList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.function.IntConsumer;
import java.util.stream.IntStream;

public abstract class LanguageModelBuiderBase  extends EmbeddingBuilderBase {
  protected static final Logger log = LoggerFactory.getLogger(CoocBasedBuilder.class.getName());

  private List<Seq> text;
  private boolean textReady = false;

  protected abstract Embedding<CharSeq> fit();

  protected List<CharSeq> dict() {
    return wordsList;
  }

  protected int textsNumber() {
    return text.size();
  }

  protected void text(int i, IntConsumer consumer) {
    final Seq seq = text.get(i);
    if (seq instanceof IntSeq) {
      ((IntSeq) seq).stream().forEach(consumer);
    }
    else throw new IllegalStateException();
  }

  protected int index(CharSequence word) {
    return wordsIndex.get(CharSeq.create(word));
  }

  protected IntSeq text(int i) {
    final Seq seq = text.get(i);
    if (seq instanceof IntSeq)
      return (IntSeq)seq;
    else throw new IllegalStateException();
  }

  protected synchronized void text(int i, IntSeq set) {
    if (i > text.size()) {
      for (int k = text.size(); k <= i; k++) {
        text.add(new IntSeq());
      }
    }
    text.set(i, set);
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
      log.info("==== Text processing phase ====");
      time = System.nanoTime();
      acquireText();
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

  protected void acquireText() throws IOException {
    /*final Path coocPath = Paths.get(this.path.getParent().toString(), strip(this.path.getFileName()) + "." + wtype().name().toLowerCase() + "-" + wleft() + "-" + wright() + "-" + minCount() + ".cooc");
    try {
      final LongSeq[] cooc = new LongSeq[wordsList.size()];
      Reader coocReader = readExisting(coocPath);
      if (coocReader != null) {
        log.info("Reading existing cooccurrences");
        CharSeqTools.llines(coocReader, true).forEach(line -> {
          final LongSeqBuilder values = new LongSeqBuilder(wordsList.size());
          final CharSeq[] wordWeightPair = new CharSeq[3];
          final CharSequence wordPairs = CharSeqTools.split(line.line, '\t')[1];
          CharSeqTools.split(wordPairs, " ", false)
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
    }*/

    if (!textReady) {
      log.info("Processing text for " + this.path);
      final long startTime = System.nanoTime();
      text = new ArrayList<>();

      try (final IntStream stream = wordsIndexesStream()) {
        TIntArrayList queue = new TIntArrayList();
        stream.forEach(idx -> {
          if (idx == Integer.MAX_VALUE) {
            text.add(new IntSeq(queue.toArray()));
            queue.resetQuick();
          } else {
            queue.add(idx);
          }
        });
      }

      log.info("Generated for " + TimeUnit.NANOSECONDS.toSeconds(System.nanoTime() - startTime) + "s");

      /*final Path coocOut = Paths.get(coocPath.toString() + ".gz");
      try (Writer coocWriter = new OutputStreamWriter(new GZIPOutputStream(Files.newOutputStream(coocOut)))) {
        log.info("Writing cooccurrences to: " + coocOut);
        for (int i = 0; i < this.cooc.size(); i++) {
          final LongSeq row = cooc(i);
          final StringBuilder builder = new StringBuilder();
          builder.append(dict().get(i)).append('\t');
          final int[] prev = new int[]{-1};
          final int finalI = i;
          row.stream().forEach(packed -> {
            final int wordId = (int)(packed >>> 32);
            if (wordId <= prev[0])
              throw new IllegalStateException(String.format("Ids in cooc for [%d] (%s) are not sorted: %d > %d", finalI, dict().get(finalI), prev[0], wordId));
            prev[0] = wordId;
            builder.append(wordId).append(':').append(dict().get(wordId)).append(':').append(CharSeqTools.ppDouble(Float.intBitsToFloat(((int)(packed & 0xFFFFFFFFL))))).append(' ');
          });
          coocWriter.append(builder, 0, builder.length() - 1).append('\n');
        }
      }
      catch (IOException ioe) {
        log.warn("Unable to write dictionary to " + coocOut, ioe);
      }*/
      textReady = true;
    }
  }
}
