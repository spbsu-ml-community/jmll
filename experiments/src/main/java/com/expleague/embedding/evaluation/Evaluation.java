package com.expleague.embedding.evaluation;

import com.expleague.commons.io.StreamTools;
import com.expleague.commons.seq.CharSeq;
import com.expleague.embedding.evaluation.metrics.CloserFurtherMetric;
import com.expleague.embedding.evaluation.metrics.WordAnalogiesMetric;
import com.expleague.ml.embedding.impl.EmbeddingImpl;

import java.io.IOException;
import java.io.InputStreamReader;
import java.io.LineNumberReader;
import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

public class Evaluation {
  public static void main(String[] args) throws IOException {
    //String file = "/Users/solar/tree/proj6_spbau/data/corpuses/text8.ss_decomp";
    String file = args[0];
    String mode = "-a";
    String metricsNames = "/home/katyakos/diploma/proj6_spbau/data/tests/text8/all_metrics_files.txt";
    String target = "/home/katyakos/diploma/proj6_spbau/data/tests/text8";
    if (!Files.exists(Paths.get(target)))
      Files.createDirectory(Paths.get(target));
    try (Reader from = Files.newBufferedReader(Paths.get(file))) {
      final EmbeddingImpl embedding = EmbeddingImpl.read(from, CharSeq.class);
      if (mode.equals("-a")) {
        WordAnalogiesMetric metric = new WordAnalogiesMetric(embedding);
        metric.measure(metricsNames, target);
      } else if (mode.equals("-cf")) {
        CloserFurtherMetric metric = new CloserFurtherMetric(embedding);
        metric.measure(metricsNames, target);
      } else if (mode.equals("-n")) {
        final QualityMetric metric = new CloserFurtherMetric(embedding);
        String input;
        LineNumberReader lnr = new LineNumberReader(new InputStreamReader(System.in));
        System.out.println("Enter your word.");
        while ((input = lnr.readLine()) != null) {
          CharSeq word = CharSeq.copy(input);
          System.out.println(metric.getIndex(word));
          List<CharSeq> result = metric.getClosestWords(word, 5);
          if (result == null) {
            System.out.println("No such word");
          } else {
            for (CharSeq ans : result)
              System.out.println("\t|" + ans);
          }
        }
      }
    }
  }
}
