package com.expleague.embedding.evaluation.metrics;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.seq.CharSeq;
import com.expleague.embedding.evaluation.QualityMetric;
import com.expleague.ml.embedding.impl.EmbeddingImpl;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;

public class CloserFurtherMetric extends QualityMetric {

  public CloserFurtherMetric(EmbeddingImpl embedding) {
    super(embedding);
  }

  @Override
  protected void check(List<String> wordsLine, int lineNumber) throws IOException {
    if (wordsLine.size() != 3) throw new IOException("There should be three words in each line." +
        String.format(" Error occurred in line number %d.", lineNumber + 1));
  }

  @Override
  public void measure(String input, String output) throws IOException {
    readMetricsNames(input);

    for (String fileName : files) {
      String short_name = readMetricsFile(fileName);
      System.out.println("Started working with " + short_name);
      File file = new File(output + "/eval_result_" + short_name);
      PrintStream fout;
      try {
        fout = new PrintStream(file);
      } catch (FileNotFoundException e) {
        throw new IOException("Couldn't find the file to write the closer-further metrics results to");
      }

      List<String> result = new ArrayList<>(words_size);
      int success = countMetric(result);
      fout.println(String.format("%d successes out of %d", success, words_size));
      fout.println();
      for (int i = 0; i < words_size; i++) {
        fout.println(result.get(i));
      }
      fout.close();
    }
  }

  private int countMetric(List<String> result) {
    int success = 0;
    for (int i = 0; i < words_size; i++) {
      if (isWordsListInVocab(words.get(i))) {
        final CharSeq w1 = words.get(i).get(0);
        final CharSeq w2 = words.get(i).get(1);
        final CharSeq w3 = words.get(i).get(2);
        final boolean suc = embedding.distance(w1, w2) > embedding.distance(w1, w3);
        result.add(resultToString(suc, w1, w2, w3));
        if (suc) success++;
      } else {
        List<CharSeq> excludes = new ArrayList<>();
        for (CharSeq word : words.get(i)) {
          if (!embedding.inVocab(word))
            excludes.add(word);
        }
        result.add("WORDS " + String.join(", ", excludes) + " ARE NOT IN VOCABULARY!");
      }
    }
    return success;
  }

  private String resultToString(boolean res, CharSeq w1, CharSeq w2, CharSeq w3) {
    if (res) {
      return String.format("TRUE: \tWord \'%s\' is closer to \'%s\' than to word \'%s\'.", w1, w2, w3);
    } else {
      return String.format("FALSE:\tWord \'%s\' is closer to \'%s\' than to word \'%s\'.", w1, w3, w2);
    }
  }
}