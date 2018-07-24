package com.expleague.joom;

import com.expleague.commons.csv.CsvTools;
import com.expleague.commons.io.StreamTools;
import com.expleague.commons.io.codec.seq.DictExpansion;
import com.expleague.commons.io.codec.seq.ListDictionary;
import com.expleague.commons.seq.*;
import com.expleague.commons.util.Pair;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.map.TObjectIntMap;
import gnu.trove.map.hash.TObjectIntHashMap;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Writer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;
import java.util.zip.GZIPInputStream;

public class ProductsSegmentation {

  public static CharSeqTools.SubstitutionCost SMART_COST = (from, to) -> {
    if (from == to)
      return 0;
    else if (Character.toUpperCase(from) == Character.toUpperCase(to))
      return 0.;
    else if (Character.isDigit(from) && Character.isDigit(to))
      return 0.5;
    else if (to == 0 && !Character.isDigit(from) && !Character.isLetter(from))
      return 0.2;
    else if (to == 0)
      return 10;
    else if (from == 0 && !Character.isDigit(to) && !Character.isLetter(to))
      return 0.2;
    else if (from == 0)
      return 0.9;
    return 10;
  };

  public static void main(String[] args) throws IOException {
    Path dictPath = Paths.get("/Users/solar/data/joom/products/products.dict");
    Path productNamesArchivePath = Paths.get("/Users/solar/data/joom/products/products.csv.gz");
    switch (args[0]) {
      case "dict": {
        List<Character> alphabet = "abcdefghijklmnopqrstuvwxyz0123456789 ".chars().mapToObj(ch -> (char) ch).collect(Collectors.toList());
        DictExpansion<Character> de = new DictExpansion<>(alphabet, 15000, System.out);
        List<CharSeq> input;
        {
          try (InputStreamReader reader = new InputStreamReader(new GZIPInputStream(Files.newInputStream(productNamesArchivePath)))) {
            input = CsvTools.csvLines(reader, new char[]{'\n', ',', '"'}).map(row -> CharSeq.intern(CharSeq.create(normalize(row.at("origName"))))).collect(Collectors.toList());
          }
        }
        for (int i = 0; i < 10; i++) {
          System.out.println("Epoch " + (i + 1));
          Collections.shuffle(input);
          input.parallelStream().forEach(de::accept);
          try (BufferedWriter writer = Files.newBufferedWriter(dictPath, StreamTools.UTF)) {
            de.print(writer);
          }
          catch (Exception ignore) {
          }
        }
        try (BufferedWriter writer = Files.newBufferedWriter(dictPath)) {
          de.print(writer);
        }
        break;
      }
      case "encode": {
        final ListDictionary<Character> dict;
        try (Stream<CharSeq> lines = CharSeqTools.lines(Files.newBufferedReader(dictPath))) {
          //noinspection unchecked
          dict = (ListDictionary<Character>) new ListDictionary(
              lines
                  .map(line -> CharSeq.create(CharSeqTools.split(line, '\t')[0]))
                  .filter(str -> str.length() > 0)
                  .<Seq<Character>>toArray(Seq[]::new)
          );
        }

        final TIntArrayList freqs;
        try (InputStreamReader reader = new InputStreamReader(new GZIPInputStream(Files.newInputStream(productNamesArchivePath)))) {
          Stream<CharSeq> charStream = CsvTools.csvLines(reader, new char[]{'\n', ',', '"'}).map(row -> CharSeq.intern(CharSeq.create(normalize(row.at("origName")))));
          freqs = buildFreqs(charStream, dict);
        }

        final int totalFreq = freqs.sum();
        try (InputStreamReader reader = new InputStreamReader(new GZIPInputStream(Files.newInputStream(productNamesArchivePath)))) {
          try (Writer writer = Files.newBufferedWriter(Paths.get("/Users/solar/data/joom/products/products.encoded"))) {
            CsvTools.csvLines(reader, new char[]{'\n', ',', '"'})
                .map(row -> CharSeq.intern(CharSeq.create(normalize(row.at("origName")))))
                .map(str -> dict.parse(str, freqs, totalFreq)).map(seq -> IntStream.of(seq.toArray()).mapToObj(Integer::toString).collect(Collectors.joining(" ")))
                .forEach(str -> {
                  try {
                    writer.write(str);
                    writer.write('\n');
                  }
                  catch (IOException e) {
                    throw new RuntimeException(e);
                  }
                });
          }
        }

        break;
      }
      case "decode": {
        final ListDictionary<Character> dict;
        try (Stream<CharSeq> lines = CharSeqTools.lines(Files.newBufferedReader(dictPath))) {
          //noinspection unchecked
          dict = (ListDictionary<Character>) new ListDictionary(
              lines
                  .map(line -> CharSeq.create(CharSeqTools.split(line, '\t')[0]))
                  .filter(str -> str.length() > 0)
                  .<Seq<Character>>toArray(Seq[]::new)
          );
        }

//        final TIntArrayList freqs;
//        try (InputStreamReader reader = new InputStreamReader(new GZIPInputStream(Files.newInputStream(productNamesArchivePath)))) {
//          Stream<CharSeq> charStream = DataTools.csvLines(reader, new char[]{'\n', ',', '"'}).map(row ->
//              CharSeq.intern(CharSeq.create(normalize(row.at("origName"))))
//          );
//
//          freqs = buildFreqs(charStream, dict);
//        }

//        final int totalFreq = freqs.sum();
        final TObjectIntMap<Pair<CharSeq, CharSeq>> codeFreqs = new TObjectIntHashMap<>();
        final Map<Pair<CharSeq, CharSeq>, CharSeq> examples = new HashMap<>();
        try (InputStreamReader reader = new InputStreamReader(new GZIPInputStream(Files.newInputStream(productNamesArchivePath)))) {
          CsvTools.csvLines(reader, new char[]{'\n', ',', '"'})
              .map(row -> CharSeq.intern(CharSeq.create(row.at("origName"))))
              .forEach(str -> {
                IntSeq parse = dict.parse(CharSeq.create(normalize(CharSeq.create(str))));//, freqs, totalFreq);
                IntStream.of(parse.toArray()).filter(idx -> idx >= 0).mapToObj(dict::get)
                    .forEach(vgram -> {
                      CharSeqBuilder builder = new CharSeqBuilder();
                      CharSeqTools.closestSubstring((CharSeq)vgram, str, SMART_COST, builder);
                      final Pair<CharSeq, CharSeq> key = Pair.create((CharSeq) vgram, builder.build());
                      codeFreqs.adjustOrPutValue(key, 1, 1);
                      examples.put(key, str);
                    });
              });
        }

        try (Writer writer = Files.newBufferedWriter(Paths.get("/Users/solar/data/joom/products/products.decoded"))) {
          for (Pair<CharSeq, CharSeq> pair : codeFreqs.keySet()) {
            writer.write(pair.first + "\t" + pair.second + "\t" + codeFreqs.get(pair) + "\t" + examples.get(pair) + "\n");
          }
        }
        catch (IOException ignore) {}

        break;
      }
    }
  }

  private static CharSequence normalize(CharSeq name) {
    String result = name.toString().toLowerCase().replaceAll("[^a-z0-9]+", " ");
    return result.endsWith(" ") ? result : result + " ";
  }

  private static <T extends Comparable<T>> TIntArrayList buildFreqs(Stream<CharSeq> stream, ListDictionary<T> dict) {
    final TIntArrayList freqs = new TIntArrayList();
    freqs.fill(0, dict.size(), 0);
    int[] stat = new int[]{0};
    stream.map(text -> {
      //noinspection unchecked,RedundantCast
      final Seq<T> seq = (Seq<T>)((Seq)text);
      if (stat[0] > 10000)
        return dict.parse(seq, freqs, stat[0]);
      else
        return dict.parse(seq);
    }).flatMapToInt(IntSeq::stream).forEach(idx -> {
      stat[0]++;
      freqs.set(idx, freqs.get(idx) + 1);
    });
    return freqs;
  }
}
