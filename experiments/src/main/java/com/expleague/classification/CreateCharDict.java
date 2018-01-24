package com.expleague.classification;

import com.expleague.commons.io.StreamTools;
import com.expleague.commons.io.codec.seq.DictExpansion;
import com.expleague.commons.io.codec.seq.Dictionary;
import com.expleague.commons.io.codec.seq.ListDictionary;
import com.expleague.commons.random.FastRandom;
import com.expleague.commons.seq.*;
import com.expleague.commons.util.ArrayTools;
import com.expleague.commons.util.ThreadTools;
import gnu.trove.list.TIntList;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.map.hash.TObjectIntHashMap;
import it.unimi.dsi.fastutil.objects.Object2IntOpenHashMap;
import org.apache.commons.compress.compressors.bzip2.BZip2CompressorInputStream;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.StringUtils;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.CharBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.function.Consumer;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * Created by Юлиан on 19.11.2015.
 */
public class CreateCharDict {

  static final String resourses = "/Users/solar/data/text_classification/";


  public static void main(String args[]) throws Exception {
    for (int i : new int[]{30000}) {
      ng20(i);
////      imdb(i);
    }
    convertNG20();
//    for (int i : new int[]{30000}) {
//      testDictionaryNG(i);
//      testDictionaryIMDB(i);
//    }
  }

  private static void ng20(int size) throws Exception {
    Path inputDir = Paths.get(resourses + "20news-bydate");

    final HashSet<Character> alpha = new HashSet<>();
    Files.find(inputDir,
        Integer.MAX_VALUE,
        (filePath, fileAttr) -> fileAttr.isRegularFile())
        .map(path -> {
          try {
            return new String(Files.readAllBytes(path), StreamTools.UTF);
          }
          catch (IOException e) {
            throw new RuntimeException(e);
          }
        })
        .forEach(line -> {
          for (int t = 0; t < line.length(); t++)
            alpha.add(line.charAt(t));
        });

    DictExpansion<Character> expansion = new DictExpansion<>(alpha, size, System.out);
    FastRandom rng = new FastRandom();
    ThreadPoolExecutor bgExecutor = ThreadTools.createBGExecutor("Parsing", 50000);
    for (int i = 0; i < 200; i++) {
      System.out.println("Iteration " + i);
      Path[] paths = Files.find(inputDir,
          Integer.MAX_VALUE,
          (filePath, fileAttr) -> fileAttr.isRegularFile())
          .toArray(Path[]::new);

      double[] rand = IntStream.range(0, paths.length).mapToDouble(c -> rng.nextDouble()).toArray();
      int[] order = ArrayTools.sequence(0, paths.length);

      ArrayTools.parallelSort(rand, order);

      CountDownLatch latch = new CountDownLatch(order.length);
      for (int k = 0; k < order.length; k++) {
        int finalK = k;
        bgExecutor.execute(() -> {
          Path path = paths[order[finalK]];
          byte[] bytes;
          try {
            bytes = Files.readAllBytes(path);
          }
          catch (IOException e) {
            throw new RuntimeException(e);
          }
          expansion.accept(CharSeq.create(new String(bytes, StreamTools.UTF)));
          latch.countDown();
        });
      }
      latch.await();
      if (i % 5 == 0) {
        final Path resultsDir = Paths.get("results");
        if (!Files.exists(resultsDir)) {
          Files.createDirectory(resultsDir);
        }
        expansion.print(new FileWriter("results/" + size + "_" + i + "_20ng.txt"));
      }
    }
    expansion.print(new FileWriter("results/" + size + "_20ng.txt"));
  }

  private static void convertNG20() throws Exception {
    List<CharSeq> alpha = new ArrayList<>();
    TIntList freqs = new TIntArrayList();
    {
      BufferedReader br = new BufferedReader(new FileReader(new File("results/30000_20ng.txt")));
      String line;
      int idx = 0;
      StringBuilder builder = new StringBuilder();
      for (int i = 0; i < 256; i++) {
        alpha.add(new CharSeqChar((char)i));

      }
      while ((line = br.readLine()) != null) {
        String[] split = line.split("\t");
        if (split.length == 2) {
          builder.append(split[0]);
          alpha.add(CharSeq.create(builder.toString()));
          freqs.add(CharSeqTools.parseInt(split[1]));
          builder = new StringBuilder();
        }
        else builder.append(line + "\n");
      }
    }
    ListDictionary<Character> dict = new ListDictionary<Character>(alpha.toArray(new CharSeq[alpha.size()]));
    Path outputDir = Paths.get(resourses + "20news-bydate-v-grams");
    Path inputDir = Paths.get(resourses + "20news-bydate");
    Files.find(inputDir,
        Integer.MAX_VALUE,
        (filePath, fileAttr) -> fileAttr.isRegularFile())
        .forEach(path -> {
          try {
            final CharSequence text = new String(Files.readAllBytes(path), "UTF-8");
            final List<CharSeq> parts = new ArrayList<>();
            final String conversion = dict.parse(CharSeq.create(text), freqs, freqs.sum()).stream()
                .peek(idx -> parts.add((CharSeq)dict.get(idx)))
                .mapToObj(Integer::toString).collect(Collectors.joining(" "));
            Path parent = path.getParent();
            String suffix = path.getName(path.getNameCount() - 1).toString();
            while (!parent.equals(inputDir)) {
              suffix = parent.getName(parent.getNameCount() - 1) + "/" + suffix;
              parent = parent.getParent();
            }
            Path out = outputDir.resolve(suffix);
            out.getParent().toFile().mkdirs();
            try (Writer wri = Files.newBufferedWriter(out)) {
                wri.append(conversion);
            };
          }
          catch (IOException e) {
            throw new RuntimeException(e);
          }
        });
  }

  private static void imdb(int size) throws Exception {

    DictExpansion<Character> expansion = new DictExpansion<>((Dictionary<Character>) Dictionary.EMPTY, size, System.out);

    Consumer<File> processFolder = folder -> {
      try {
        for (File file : folder.listFiles()) {
          BufferedReader reader = new BufferedReader(new FileReader(file));
          String line;
          while ((line = reader.readLine()) != null) {
            expansion.accept(CharSeq.create(line));
          }
        }
      }
      catch (Exception ex) {
        ex.printStackTrace();
      }
    };

    for (int i = 0; i < 200; i++) {
      System.out.println("Iteration " + i);

      processFolder.accept(new File(resourses + "aclImdb\\train\\neg"));
      processFolder.accept(new File(resourses + "aclImdb\\train\\pos"));
      processFolder.accept(new File(resourses + "aclImdb\\train\\unsup"));
      if (i % 5 == 0)
        expansion.print(new FileWriter("results/" + size + "_" + i + "_imdb.txt"));
    }
    expansion.print(new FileWriter("results/" + size + "_imdb.txt"));
  }

  public static void testDictionaryNG(int size) throws Exception {

    HashMap<String, Double> map = new HashMap<>();
    {
      BufferedReader br = new BufferedReader(new FileReader(new File("results/" + size + "_20ng"
          + ".txt")));
      String line;
      while ((line = br.readLine()) != null) {
        if (line.split("\t").length == 2)
          map.put(line.split("\t")[0], 1. * line.split("\t")[0].length() - 1);
      }
    }
    SimpleSplitter simpleSplitter = new SimpleSplitter(map.keySet());
    WeighedSplitter weighedSplitter = new WeighedSplitter(map);

    String collections[] = new String[]{"20ng"};
    for (String collection : collections) {
      System.out.println(size);
      System.out.println("Collection: " + collection);
      Rocchio usual = new Rocchio(simpleSplitter::split);
      Rocchio extended = new Rocchio(weighedSplitter::split);
      BufferedReader reader = new BufferedReader(new InputStreamReader(new
          BZip2CompressorInputStream(new FileInputStream(resourses + "20ng-train.txt.bz2"))));
      String line;
      while ((line = reader.readLine()) != null) {
        String cls = line.split("\t")[0];
        String body = line.split("\t")[1];
        usual.addDocument(body, cls);
        extended.addDocument(body, cls);
      }
      usual.buildClassifier();
      extended.buildClassifier();
      reader = new BufferedReader(new InputStreamReader(new BZip2CompressorInputStream(new
          FileInputStream(resourses + "20ng-test.txt.bz2"))));

      int tests = 0;
      int usualCorrect = 0;
      int extendedCorrect = 0;
      int similar = 0;

      while ((line = reader.readLine()) != null) {
        String cls = line.split("\t")[0];
        String body = line.split("\t")[1];
        String usualPrediction = usual.classify(body);
        String extendedPrediction = extended.classify(body);
        tests++;
        if (usualPrediction.equals(cls))
          usualCorrect++;
        if (extendedPrediction.equals(cls))
          extendedCorrect++;
        if (usualPrediction.equals(extendedPrediction))
          similar++;
      }
      System.out.println("Used : " + extended.usedSize());
      System.out.println("Extended dictionary correct : " + 1. * extendedCorrect / tests);
      System.out.println("Usual dictionary correct : " + 1. * usualCorrect / tests);
      System.out.println("The same answer : " + 1. * similar / tests + "\n");
      System.out.println("===================================================");
    }
  }


  public static void testDictionaryIMDB(int size) throws Exception {
    HashMap<String, Double> map = new HashMap<>();
    {
      BufferedReader br = new BufferedReader(new FileReader(new File("results/" + size + "_imdb" +
          ".txt")));
      String line;
      while ((line = br.readLine()) != null) {
        if (line.split("\t").length == 2)
          map.put(line.split("\t")[0], 1. * line.split("\t")[0].length() - 1);

      }
    }
    SimpleSplitter simpleSplitter = new SimpleSplitter(map.keySet());
    WeighedSplitter weighedSplitter = new WeighedSplitter(map);

    Rocchio usual = new Rocchio(simpleSplitter::split);
    Rocchio extended = new Rocchio(weighedSplitter::split);
    for (String cls : new String[]{"neg", "pos"})
      for (File file : new File(resourses + "aclImdb\\train\\" + cls).listFiles()) {
        BufferedReader reader = new BufferedReader(new FileReader(file));
        String line;
        StringBuilder doc = new StringBuilder();
        while ((line = reader.readLine()) != null) {
          doc.append(line);
        }
        usual.addDocument(doc.toString(), cls);
        extended.addDocument(doc.toString(), cls);
      }

    usual.buildClassifier();
    extended.buildClassifier();

    int tests = 0;
    int usualCorrect = 0;
    int extendedCorrect = 0;
    int similar = 0;

    for (String cls : new String[]{"pos", "neg"})
      for (File file : new File(resourses + "aclImdb\\test\\" + cls).listFiles()) {
        BufferedReader reader = new BufferedReader(new FileReader(file));
        String line;
        StringBuilder doc = new StringBuilder();
        while ((line = reader.readLine()) != null) {
          doc.append(line);
        }
        String body = doc.toString();
        String usualPrediction = usual.classify(body);
        String extendedPrediction = extended.classify(body);
        tests++;
        if (usualPrediction.equals(cls))
          usualCorrect++;
        if (extendedPrediction.equals(cls))
          extendedCorrect++;
        if (usualPrediction.equals(extendedPrediction))
          similar++;
      }


    System.out.println("Used : " + extended.usedSize());
    System.out.println("Extended dictionary correct : " + 1. * extendedCorrect / tests);
    System.out.println("Usual dictionary correct : " + 1. * usualCorrect / tests);
    System.out.println("The same answer : " + 1. * similar / tests + "\n");
    System.out.println("===================================================");

  }


  public static void createCSV() throws Exception {
    HashMap<String, Double> map = new HashMap<>();
    {
      BufferedReader br = new BufferedReader(new FileReader(new File("results/"  + "30000_20ng" +
          ".txt")));
      String line;
      while((line = br.readLine()) != null){
        if(line.split("\t").length == 2)
          map.put(line.split("\t")[0], 1.*line.split("\t")[0].length() - 1);

      }
    }
    WeighedSplitter weighedSplitter = new WeighedSplitter(map);
    String collections[] = new String[]{
        "20ng"
    };
    Object2IntOpenHashMap<String> indexes = new Object2IntOpenHashMap<>();
    for(String collection : collections) {
      StringBuilder header = new StringBuilder("class");
      {
        System.out.println(1);
        BufferedReader reader = new BufferedReader(new InputStreamReader(new BZip2CompressorInputStream(new FileInputStream(
            resourses + "20ng-train.txt.bz2"
        ))));
        String line;
        while((line = reader.readLine()) != null){
          String body = line.split("\t")[1];
          String[] words = weighedSplitter.split(body);
          for(String w : words){
            if(!indexes.containsKey(w)) {
              indexes.put(w, indexes.size());
              header.append(";").append(w);
            }
          }
        }
      }
      {
        System.out.println(2);
        BufferedReader reader = new BufferedReader(new InputStreamReader(new BZip2CompressorInputStream(new FileInputStream(
            resourses + "20ng-test.txt.bz2"
        ))));
        String line;
        while((line = reader.readLine()) != null){
          String body = line.split("\t")[1];
          String[] words = weighedSplitter.split(body);
          for(String w : words){
            if(!indexes.containsKey(w)) {
              indexes.put(w, indexes.size());
              header.append(";").append(w);
            }
          }
        }
      }

      {
        System.out.println(3);
        BufferedReader reader = new BufferedReader(new InputStreamReader(new BZip2CompressorInputStream(new FileInputStream(
            resourses + "20ng-train.txt.bz2"
        ))));
        String line;
        PrintWriter pw = new PrintWriter("results/train.txt");
        pw.println(header);
        while((line = reader.readLine()) != null){
          String cls = line.split("\t")[0];
          String body = line.split("\t")[1];
          String[] words = weighedSplitter.split(body);
          int[] tf = new int[indexes.size()];
          for(String w : words)
            tf[indexes.getInt(w)]++;
          pw.print(cls + ';');
          pw.println(StringUtils.join(ArrayUtils.toObject(tf), ";"));
        }
        pw.close();
      }
      {
        System.out.println(4);

        BufferedReader reader = new BufferedReader(new InputStreamReader(new BZip2CompressorInputStream(new FileInputStream(
            resourses + "20ng-test.txt.bz2"
        ))));
        String line;
        PrintWriter pw = new PrintWriter("results/test.txt");
        pw.println(header);
        while((line = reader.readLine()) != null){
          String cls = line.split("\t")[0];
          String body = line.split("\t")[1];
          String[] words = weighedSplitter.split(body);
          int[] tf = new int[indexes.size()];
          for(String w : words)
            tf[indexes.getInt(w)]++;
          pw.print(cls + ';');
          pw.println(StringUtils.join(ArrayUtils.toObject(tf), ";"));
        }
        pw.close();
      }
    }
  }


}
