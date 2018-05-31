package com.expleague.de;

import com.expleague.commons.io.StreamTools;
import com.expleague.commons.io.codec.seq.DictExpansion;
import com.expleague.commons.io.codec.seq.ListDictionary;
import com.expleague.commons.seq.*;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.map.hash.TIntIntHashMap;
import org.apache.commons.cli.*;
import org.jetbrains.annotations.NotNull;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.*;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class DictTools {
  private static String normalize(String s) {
    //        return s.toLowerCase().replaceAll("\\W|\\d", "");
    String result = s.toLowerCase().replaceAll("[^a-zA-Z0-9]", "");
    //              .replaceAll("\\*\\*+", " ")
    //              .replaceAll("--+", " ")
    //              .replaceAll("[\\.,:!?;\\\\/$]", "")
    //              .replaceAll("['\"<>()\\[\\]{};+\n\r]", " ")
    //              .replaceAll("\\s+", " ");
    //      System.out.println(result);
    return result;
  }

  private static <T extends Comparable<T>> void buildDictionary(final Path dir, int bits, final int dictSize, final int iterNum) throws IOException {
    final Set<T> allCharacters = new HashSet<>();
    final List<Seq<T>> content = new ArrayList<>();
    DictTools.<T>getDirContentStream(dir, bits).forEach(content::add);
    content.forEach(text -> text.forEach(allCharacters::add));
    final DictExpansion<T> expansion = new DictExpansion<>(allCharacters, dictSize, System.out);
    for (int i = 0; i < iterNum; i++) {
      List<Seq<T>> nextIter = new ArrayList<>(content);
      Collections.shuffle(nextIter);
      IntStream.range(0, content.size()).parallel().forEach(idx -> expansion.accept(nextIter.get(idx)));
      System.out.println(i + "-th iter end");
    }

    final Path dictPath = dictPath(dir, bits, dictSize);
    System.out.println("writing dict to " + dictPath);
    try (BufferedWriter writer = Files.newBufferedWriter(dictPath)){
      expansion.print(writer);
    }
    catch (IOException ioe) {
      ioe.printStackTrace();
    }
    System.out.println("END");
  }

  private static <T extends Comparable<T>> void transferDirToBow(int bits, ListDictionary<T> dict, TIntArrayList freqs, int totalFreq, Path sourceDir, Path dstDir) throws IOException {
    Files.walk(sourceDir, Integer.MAX_VALUE, FileVisitOption.FOLLOW_LINKS).filter(file -> !Files.isDirectory(file)).forEach(file -> {
      try {
        final TIntIntHashMap tf = new TIntIntHashMap();
        final String relative = file.toString().substring(sourceDir.toString().length() + 1);
        final Path dst = dstDir.resolve(relative + ".bow");
        if (!Files.exists(dst.getParent()))
          Files.createDirectories(dst.getParent());

        final Seq<T> seq = convertToSeq(file, bits, Charset.forName("Latin1"));
        dict.parse(seq, freqs, totalFreq).stream()
            .forEach(id -> tf.adjustOrPutValue(id, 1, 1));

        Files.write(dst,
            IntStream.of(tf.keys()).mapToObj(id -> id + "\t" + tf.get(id) + "\t" + dict.get(id)).collect(Collectors.toList()),
            StandardOpenOption.WRITE, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING
        );
      }
      catch (IOException e) {
        throw new RuntimeException(e);
      }
    });
  }

  private static Options options = new Options();
  static {
    options.addOption(Option.builder("b").longOpt("bits").desc("Bit length of tuple").hasArg().build());
    options.addOption(Option.builder("n").longOpt("size").desc("Size of dictionary").hasArg().build());
    options.addOption(Option.builder("i").longOpt("iterations").desc("Number of cycles through the data to build dictionary").hasArg().build());
  }

  public static <T extends Comparable<T>> void main(String... args) throws IOException, ParseException {
    final CommandLineParser parser = new DefaultParser();
    final CommandLine cliOptions = parser.parse(options, args);
    String mode = "apply";
    List<String> freeArgs = new ArrayList<>(Arrays.asList(cliOptions.getArgs()));
    if (freeArgs.size() > 0 && !freeArgs.get(0).contains("/"))
      mode = freeArgs.remove(0);

    final Path dir2process = Paths.get(freeArgs.size() > 0 ? freeArgs.get(0) : "./");
    int bits = Integer.parseInt(cliOptions.getOptionValue("b", "-1"));
    int dictSize = Integer.parseInt(cliOptions.getOptionValue("n", "15000"));
    int iterNum = Integer.parseInt(cliOptions.getOptionValue("i", "30"));

    long heapSize = Runtime.getRuntime().maxMemory();
    System.out.println("Heap Size = " + heapSize / 1024 / 1024 + "Mb");
    final Path dictionary = dir2process.resolve("dict_" + dictSize + ".dict");

    if ("build".equals(mode) || !Files.exists(dictionary)) {
      System.out.println("Building dictionary " + dictionary);
      buildDictionary(dir2process, bits, dictSize, iterNum);
    }
    else System.out.println("Using existing dictionary: " + dictionary);

    if ("apply".equals(mode) || "build".equals(mode)) {
      final boolean binary = bits > 0 && bits <= 8;
      //noinspection unchecked
      final ListDictionary<T> dict = (ListDictionary<T>) new ListDictionary(
          CharSeqTools.lines(Files.newBufferedReader(dictionary))
              .map(line -> binary ? ByteSeq.create(CharSeqTools.split(line, '\t')[0].toString()) : CharSeq.create(CharSeqTools.split(line, '\t')[0]))
              .filter(str -> str.length() > 0)
              .<Seq<Character>>toArray(Seq[]::new)
      );

      final TIntArrayList freqs = buildFreqs(dir2process, bits, dict);
      final int power = freqs.sum();
      Files.list(dir2process).filter(Files::isDirectory).filter(dir -> !dir.toString().endsWith("-bows")).forEach(sub -> {
        try {
          System.out.println("Converting directory: " + sub.toString());
          final Path dstDir = Paths.get(sub.toString() + "-bows");
          transferDirToBow(bits, dict, freqs, power, sub, dstDir);
        }
        catch (IOException e) {
          throw new RuntimeException(e);
        }
      });
    }
  }

  private static <T extends Comparable<T>> TIntArrayList buildFreqs(Path dir2process, int bits, ListDictionary<T> dict) throws IOException {
    final TIntArrayList freqs = new TIntArrayList();
    freqs.fill(0, dict.size(), 0);
    int[] stat = new int[]{0};
    getDirContentStream(dir2process, bits).map(text -> {
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

  // inner routines

  private static <T extends Comparable<T>> Path dictPath(Path dir, int bits, int size) {
    return Paths.get(dir + "/dict_" + (bits > 0 ? bits + "_" : "") + size + ".dict");
  }

  @NotNull
  private static <T> Stream<Seq<T>> getDirContentStream(Path dir, int bits) throws IOException {
    return Files.walk(dir, Integer.MAX_VALUE, FileVisitOption.FOLLOW_LINKS)
        .filter(file -> !Files.isDirectory(file)).filter(file -> !file.toString().endsWith(".dict")).filter(file -> !file.toString().contains("-bows"))
        .map(file -> convertToSeq(file, bits, Charset.forName("Latin1")));
  }

  private static <T> Seq<T> convertToSeq(Path file, int bits, Charset charset) {
    try {
      if (bits > 0 && bits <= 8) {
        //noinspection unchecked
        return (Seq<T>) new ByteSeq(changeByteSize(Files.readAllBytes(file), bits));
      }
      else {
        try (BufferedReader reader = Files.newBufferedReader(file, charset)) {
          //noinspection unchecked
          return (Seq<T>) CharSeq.create(normalize(StreamTools.readReader(reader).toString()));
        }
      }
    }
    catch (IOException e) {
      throw new RuntimeException(e);
    }

  }

  private static byte[] changeByteSize(byte[] bytes, int byteSize) {
    if (byteSize == 8) {
      return bytes;
    }
    StringBuilder sb = new StringBuilder();
    int offset = byteSize - (bytes.length * 8) % byteSize;
    for (int i = 0; i < offset; i++) {
      sb.append('0');
    }
    for (byte b : bytes) {
      for (int j = 0; j < 8; j++) {
        if ((b & (1 << 7)) == 0) {
          sb.append('0');
        } else {
          sb.append('1');
        }
        b <<= 1;
      }
    }
    String byteStr = sb.toString();
    byte[] newBytes = new byte[byteStr.length() / byteSize];
    int k = 0;
    for (int i = 0; i < newBytes.length; i++) {
      byte b = 0;
      int j = 0;
      while (j < byteSize) {
        b <<= 1;
        if (byteStr.charAt(k) == '1') {
          b |= 1;
        }
        j++;
        k++;
      }

      newBytes[i] = b;
    }
    return newBytes;
  }
}