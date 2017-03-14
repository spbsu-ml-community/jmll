package com.spbsu.direct;

import com.spbsu.commons.func.Action;
import com.spbsu.commons.io.StreamTools;
import com.spbsu.commons.io.codec.seq.DictExpansion;
import com.spbsu.commons.io.codec.seq.ListDictionary;
import com.spbsu.commons.math.io.Vec2CharSequenceConverter;
import com.spbsu.commons.math.vectors.impl.vectors.SparseVec;
import com.spbsu.commons.seq.*;
import com.spbsu.commons.util.ThreadTools;
import com.spbsu.direct.gen.SimpleGenerativeModel;
import gnu.trove.list.TIntList;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.map.TObjectDoubleMap;
import gnu.trove.map.custom_hash.TObjectDoubleCustomHashMap;
import gnu.trove.strategy.HashingStrategy;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

/**
 * User: solar
 * Date: 07.10.15
 * Time: 15:56
 */
public class BroadMatch {
  public static boolean debug = true;

  volatile static int index = 0;
  volatile static int windex = 0;

  public static void main(String[] args) throws IOException {
    if (args.length < 2)
      throw new IllegalArgumentException("Need at least two arguments: mode and file to work with");
    switch (args[0]) {
      case "-dict": {
        final DictExpansion<CharSeq> expansion = new DictExpansion<>(Integer.parseInt(args[1]), System.out);
        final String outputFile = args[2];
        final Action<DictExpansion<CharSeq>> printer = new Action<DictExpansion<CharSeq>>() {
          int dictIndex = 0;

          @Override
          public void invoke(DictExpansion<CharSeq> result) {
            try {
              System.out.println("Dump dictionary #" + dictIndex);
              result.print(new FileWriter(StreamTools.stripExtension(outputFile) + "-" + dictIndex + ".dict"));
              dictIndex++;
              windex = 0;
            } catch (Exception e) {
              e.printStackTrace();
            }
          }
        };
        expansion.addListener(printer);
        final ThreadPoolExecutor executor = ThreadTools.createBGExecutor("Creating DictExpansion", 100000);
        for (int i = 3; i < args.length; i++) {
          CharSeqTools.processLines(StreamTools.openTextFile(args[i]), new Action<CharSequence>() {
            String current;

            @Override
            public void invoke(CharSequence line) {
              final CharSequence[] parts = new CharSequence[3];
              if (CharSeqTools.split(line, '\t', parts).length != 3)
                throw new IllegalArgumentException("Each input line must contain <uid>\\t<ts>\\t<query> triplet. This one: [" + line + "]@" + outputFile + ":" + index + " does not.");
              if (CharSeqTools.startsWith(parts[0], "uu/") || CharSeqTools.startsWith(parts[0], "r"))
                return;
              final String uid = parts[0].toString();
              final String query = parts[2].toString();
              if (query.equals(current))
                return;
              current = query;
//          if (!CharSeqTools.equals(parts[0], currentUser))
              final Runnable item = () -> {
                final String normalizedQuery = normalizeQuery(query);
                final ArraySeq<CharSeq> seq = convertToSeq(normalizedQuery);
                if (windex++ < 10)
                  System.out.println(uid + ": " + normalizedQuery + " -> " + seq);
                expansion.accept(seq);
              };
              final BlockingQueue<Runnable> queue = executor.getQueue();
              //noinspection Duplicates
              if (queue.remainingCapacity() == 0) {
                try {
                  queue.put(item);
                } catch (InterruptedException e) {
                  throw new RuntimeException(e);
                }
              } else executor.execute(item);
            }
          });
        }
        break;
      }
      case "-depends": {
        final double alpha = 0.5;
        final TIntList freqsLA = new TIntArrayList();
        final ListDictionary<CharSeq> dict = loadDictionaryWithFreqs(args[1], freqsLA);

        final SimpleGenerativeModel model = new SimpleGenerativeModel(dict, freqsLA);
        model.loadStatistics(args[2]);

        for (int i = 3; i < args.length; i++) {
          final String fileName = args[i];
          CharSeqTools.processLines(StreamTools.openTextFile(fileName), new Action<CharSequence>() {
            long ts;
            String query;
            String user;
            IntSeq prevQSeq;

            @Override
            public void invoke(CharSequence line) {
              final CharSequence[] parts = new CharSequence[3];
              if (CharSeqTools.split(line, '\t', parts).length != 3)
                throw new IllegalArgumentException("Each input line must contain <uid>\\t<ts>\\t<query> triplet. This one: [" + line + "]@" + fileName + ":" + index + " does not.");
              if (CharSeqTools.startsWith(parts[0], "uu/") || CharSeqTools.startsWith(parts[0], "r"))
                return;
              final long ts = CharSeqTools.parseLong(parts[1]);
              final String query = normalizeQuery(parts[2].toString());
              if (query == null || query.equals(this.query)) {
                this.ts = ts;
                return;
              }
              final IntSeq currentQSeq = dropUnknown(dict.parse(convertToSeq(query), model.freqs, model.totalFreq));
              if (currentQSeq == null) {
                this.ts = ts;
                return;
              }
              model.processSeq(currentQSeq);
              final String prev = parts[0].equals(this.user) && ts - this.ts < TimeUnit.MINUTES.toSeconds(30) ? this.query : null;
              this.query = parts[2].toString();
              this.user = parts[0].toString();
              this.ts = ts;
              if (prev != null && prevQSeq != null) {
                model.processGeneration(prevQSeq, currentQSeq, alpha);
              }
              prevQSeq = currentQSeq;
              if (++index % 10000000 == 0) {
                try (final Writer out = new OutputStreamWriter(new FileOutputStream("output-" + (index / 10000000) + ".txt"))) {
                  model.print(out, true);
                } catch (IOException e) {
                  e.printStackTrace();
                }
              }
            }
          });
        }
        try (final Writer out = new OutputStreamWriter(new FileOutputStream("output-" + (index / 10000000) + ".txt"))) {
          model.print(out, false);
        } catch (IOException e) {
          e.printStackTrace();
        }
        break;
      }
      case "-stats": {
        final Vec2CharSequenceConverter converter = new Vec2CharSequenceConverter();
        final TIntList freqs = new TIntArrayList();
        final ListDictionary<CharSeq> dict = loadDictionaryWithFreqs(args[1], freqs);
        final SparseVec[] stats = new SparseVec[dict.size() + 1];

        for (int i = 0; i < stats.length; i++) {
          stats[i] = new SparseVec(dict.size());
        }

        final String outputFile = args[2];
        //noinspection LoopStatementThatDoesntLoop
        for (int i = 3; i < args.length; i++) {
          CharSeqTools.processLines(StreamTools.openTextFile(args[i]), new Action<CharSequence>() {
            long ts;
            String query;
            String user;
            IntSeq prevQSeq;
            double totalFreq = freqs.sum();

            @Override
            public void invoke(CharSequence line) {
              final CharSequence[] parts = new CharSequence[3];
              if (CharSeqTools.split(line, '\t', parts).length != 3)
                throw new IllegalArgumentException("Each input line must contain <uid>\\t<ts>\\t<query> triplet. This one: [" + line + "]@" + args[i] + ":" + index + " does not.");
              if (CharSeqTools.startsWith(parts[0], "uu/") || CharSeqTools.startsWith(parts[0], "r"))
                return;
              final long ts = CharSeqTools.parseLong(parts[1]);
              final String query = normalizeQuery(parts[2].toString());

              if (query == null || query.equals(this.query)) {
                this.ts = ts;
                return;
              }
              final IntSeq currentQSeq = dropUnknown(dict.parse(convertToSeq(query), freqs, totalFreq));
              if (currentQSeq == null) {
                prevQSeq = null;
                this.query = null;
                return;
              }

              for (int i = 0; i < currentQSeq.length(); i++) {
                final int symbol = currentQSeq.intAt(i);
                if (symbol >= freqs.size())
                  freqs.fill(freqs.size(), symbol + 1, 0);
                freqs.set(symbol, freqs.get(symbol) + 1);
              }

              final CharSequence uid = parts[0];
              if (!uid.equals(this.user)) {
                { // session start
                  prevQSeq = null;
                  for (int i = 0; i < currentQSeq.length(); i++) {
                    stats[dict.size()].adjust(currentQSeq.intAt(i), 1.);
                  }
                }
              }
              final IntSeq prevQSeq = uid.equals(this.user) && ts - this.ts < TimeUnit.MINUTES.toSeconds(30) ? this.prevQSeq : null;
              this.query = query;
              this.user = uid.toString();
              this.ts = ts;
              if (prevQSeq != null) {
                for (int i = 0; i < prevQSeq.length(); i++) {
                  for (int j = 0; j < currentQSeq.length(); j++) {
                    stats[prevQSeq.intAt(i)].adjust(currentQSeq.intAt(j), 1.);
                  }
                }
              }
              this.prevQSeq = currentQSeq;

              { // stats dump
                if (++index % 10000000 == 0) {
                  final String outputFileI = StreamTools.stripExtension(outputFile) + "-" + (index / 10000000) + ".stats";
                  System.out.println("Dump " + outputFileI);
                  try (final Writer out = new OutputStreamWriter(new FileOutputStream(outputFileI))) {
                    for (int i = 0; i < stats.length; i++) {
                      final SparseVec stat = stats[i];
                      if (i < dict.size())
                        out.append(dict.get(i).toString());
                      else
                        out.append(SimpleGenerativeModel.EMPTY_ID);
                      out.append("\t");
                      out.append(converter.convertTo(stat));
                      out.append("\n");
                    }
                  } catch (IOException e) {
                    e.printStackTrace();
                  }
                }
              }
            }
          });
          break;
        }
      }
      case "-query": {
        final TIntList freqs = new TIntArrayList();
        final ListDictionary<CharSeq> dict = loadDictionaryWithFreqs(args[1], freqs);
        final SimpleGenerativeModel model = new SimpleGenerativeModel(dict, freqs);
        model.load(args[2]);
        CharSeqTools.processLines(new InputStreamReader(System.in, StreamTools.UTF), (Action<CharSequence>) arg -> {
          String query = arg.toString();
          normalizeQuery(query);
          System.out.println(model.findTheBestExpansion(convertToSeq(normalizeQuery(arg.toString()))));
        });
      }
    }
  }

  @Nullable
  private static String normalizeQuery(String query) {
    query = query.replaceAll("[;,.:\\(\\)\"\'«»!\\]\\[\\{\\}<>]", "");
    query = query.replaceAll("\\s+", " ");
    return query.toLowerCase();
  }

  @NotNull
  private static ListDictionary<CharSeq> loadDictionaryWithFreqs(String arg, final TIntList freqs) throws IOException {
    final ListDictionary<CharSeq> dict;
    { // dict
      final TObjectDoubleMap<Seq<CharSeq>> freqsHash = new TObjectDoubleCustomHashMap<>(new HashingStrategy<Object>() {
        @Override
        public int computeHashCode(Object object) {
          return object.hashCode();
        }

        @Override
        public boolean equals(Object o1, Object o2) {
          return o1.equals(o2);
        }
      });
      final List<Seq<CharSeq>> dictSeqs = new ArrayList<>();

      CharSeqTools.processLines(StreamTools.openTextFile(arg), (Action<CharSequence>) line -> {
        final CharSequence[] split = CharSeqTools.split(line, '\t', new CharSequence[2]);
        final CharSequence[] parts = CharSeqTools.split(split[0].subSequence(1, split[0].length() - 1), ", ");
        final SeqBuilder<CharSeq> builder = new ArraySeqBuilder<>(CharSeq.class);
        for (final CharSequence part : parts) {
          builder.add(CharSeq.create(part.toString()));
        }
        final Seq<CharSeq> seq = builder.build();
        dictSeqs.add(seq);
        freqsHash.put(seq, CharSeqTools.parseDouble(split[1]));
      });
      //noinspection unchecked
      dict = new ListDictionary<>(dictSeqs.toArray(new Seq[dictSeqs.size()]));
      freqs.fill(0, dict.size(), 0);
      for (int i = 0; i < dict.size(); i++) {
        final double val = freqsHash.get(dict.get(i));
        freqs.add((int)val);
      }
    }
    return dict;
  }

  @Nullable
  private static IntSeq dropUnknown(IntSeq parse) {
    final IntSeqBuilder builder = new IntSeqBuilder();
    for (int i = 0; i < parse.length(); i++) {
      final int val = parse.intAt(i);
      if (val >= 0)
        builder.add(val);
    }
    return builder.length() > 0 ? builder.build() : null;
  }

  private static ArraySeq<CharSeq> convertToSeq(CharSequence word) {
    final CharSeq[] words = new CharSeq[100];
    final String query = word.toString();
    final int wcount = CharSeqTools.trySplit(CharSeq.create(query), ' ', words);
    return new ArraySeq<>(words, 0, wcount);
  }
}
