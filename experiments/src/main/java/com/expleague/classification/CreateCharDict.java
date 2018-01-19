package com.expleague.classification;

import com.expleague.commons.io.codec.seq.DictExpansion;
import com.expleague.commons.io.codec.seq.Dictionary;
import com.expleague.commons.seq.CharSeqAdapter;
import it.unimi.dsi.fastutil.objects.Object2IntOpenHashMap;
import org.apache.commons.compress.compressors.bzip2.BZip2CompressorInputStream;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.StringUtils;

import java.io.*;
import java.util.*;
import java.util.function.Consumer;

/**
 * Created by Юлиан on 19.11.2015.
 */
public class CreateCharDict {

  static final String resourses = "D:\\experiments\\compression\\";


  public static void main(String args[]) throws Exception {
//    for (int i : new int[]{1000, 5000, 10000, 30000, 50000, 100000}) {
//      ng20(i);
//      imdb(i);
//    }
    for (int i : new int[]{1000, 5000, 10000, 30000, 50000, 100000}) {
      testDictionaryNG(i);
      testDictionaryIMDB(i);
    }
  }

  private static void ng20(int size) throws Exception {

    DictExpansion<Character> expansion = new DictExpansion<>((Dictionary<Character>) Dictionary
        .EMPTY, size, System.out);
    for (int i = 0; i < 200; i++) {
      System.out.println("Iteration " + i);
      BufferedReader reader = new BufferedReader(new InputStreamReader(new
          BZip2CompressorInputStream(new FileInputStream(resourses + "20ng-train.bz2"))));
      String line;
      while ((line = reader.readLine()) != null) {
        String[] parts = line.split("\t");
        if (parts.length > 1)
          expansion.accept(new CharSeqAdapter(parts[1]));
      }
      if (i % 5 == 0)
        expansion.print(new FileWriter("results/" + size + "_" + i + "_20ng.txt"));
    }
    expansion.print(new FileWriter("results/" + size + "_20ng.txt"));

  }

  private static void imdb(int size) throws Exception {

    DictExpansion<Character> expansion = new DictExpansion<>((Dictionary<Character>) Dictionary
        .EMPTY, size, System.out);

    Consumer<File> processFolder = folder -> {
      try {
        for (File file : folder.listFiles()) {
          BufferedReader reader = new BufferedReader(new FileReader(file));
          String line;
          while ((line = reader.readLine()) != null) {
            expansion.accept(new CharSeqAdapter(line));
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
