package com.spbsu.direct;

import com.spbsu.commons.func.Action;
import com.spbsu.commons.io.StreamTools;
import com.spbsu.commons.io.codec.seq.ListDictionary;
import com.spbsu.commons.seq.*;
import com.spbsu.commons.text.StringUtils;
import gnu.trove.list.TIntList;
import gnu.trove.map.TObjectDoubleMap;
import gnu.trove.map.custom_hash.TObjectDoubleCustomHashMap;
import gnu.trove.strategy.HashingStrategy;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.io.IOException;
import java.util.*;


public final class Utils {
  private static final int SEQ_SIZE = 100;

  /**
   * Class is responsible for time measurements
   * Use it in a single thread!
   */
  public static final class Timer {
    static Stack<Long> times = new Stack<>();
    static Map<String, Double> durations = new HashMap<>();

    private static String indent() {
      return StringUtils.repeatWithDelimeter("", "\t", times.size());
    }

    public static void start(final String tag, final boolean show) {
      if (tag != null && show) {
        System.out.println(indent() + tag);
      }

      times.push(System.currentTimeMillis());
    }

    public static void stop(final String tag, final boolean show) {
      final double duration = (System.currentTimeMillis() - times.pop()) / 1000.0;

      if (tag != null) {
        durations.put(tag, durations.getOrDefault(tag, 0.0) + duration);

        if (show) {
          System.out.println(String.format("%s%s: %.3fs", indent(), tag, duration));
        }
      } else {
        durations.put("other", durations.getOrDefault("other", 0.0) + duration);

        if (show) {
          System.out.println(String.format("%s%.3fs", indent(), duration));
        }
      }
    }

    public static void showStatistics(final String relativeTag) {
      System.out.println("\nStatistics:");
      durations.entrySet().forEach(it -> {
        if (relativeTag != null) {
          System.out.println(String.format("%s: %.3fs (%.3f%%)", it.getKey(), it.getValue(), 100 * it.getValue() / durations.get(relativeTag)));
        } else {
          System.out.println(String.format("%s: %.3fs", it.getKey(), it.getValue()));
        }
      });
      System.out.println();
    }

    public static void clearStatistics() {
      durations.clear();
    }
  }

  @Nullable
  public static IntSeq dropUnknown(IntSeq parse) {
    final IntSeqBuilder builder = new IntSeqBuilder();
    Arrays.stream(parse.arr).filter(val -> val >= 0).forEach(builder::add);
    return builder.length() > 0 ? builder.build() : null;
  }

  @NotNull
  public static ArraySeq<CharSeq> convertToSeq(CharSequence word) {
    final CharSeq[] words = new CharSeq[SEQ_SIZE];
    final int wordsCount = CharSeqTools.trySplit(CharSeq.create(word.toString()), ' ', words);
    return new ArraySeq<>(words, 0, wordsCount);
  }

  @Nullable
  public static String normalizeQuery(String query) {
    query = query.replaceAll("[;,.:\\(\\)\"\'«»!\\]\\[\\{\\}<>]", "");
    query = query.replaceAll("\\s+", " ");
    return query.toLowerCase();
  }

  @NotNull
  public static ListDictionary<CharSeq> loadDictionaryWithFreqs(String arg, final TIntList freqs) throws IOException {
    final ListDictionary<CharSeq> dict;

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
      freqs.add((int) freqsHash.get(dict.get(i)));
    }

    return dict;
  }
}
