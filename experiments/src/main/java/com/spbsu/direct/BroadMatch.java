package com.spbsu.direct;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectWriter;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.spbsu.commons.func.Action;
import com.spbsu.commons.io.StreamTools;
import com.spbsu.commons.io.codec.seq.DictExpansion;
import com.spbsu.commons.io.codec.seq.Dictionary;
import com.spbsu.commons.math.vectors.VecIterator;
import com.spbsu.commons.math.vectors.impl.vectors.SparseVec;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.seq.*;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.commons.util.Holder;
import com.spbsu.commons.util.ThreadTools;

import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.zip.GZIPInputStream;

import static java.lang.Math.exp;
import static java.lang.Math.log;

/**
 * User: solar
 * Date: 07.10.15
 * Time: 15:56
 */
public class BroadMatch {
  public static class WordGenProbabilityProvider {
    private final SparseVec currentScores;
    private double denominator;
    private double undefined;

    public WordGenProbabilityProvider(int dim, int windex) {
      currentScores = new SparseVec(dim);
      currentScores.set(windex, 1.);
      undefined = 0;
      denominator = dim + 1; // 1 + \sum_1^dim e^0
    }

    public double p(int windex) {
      if (windex < currentScores.dim())
        return (exp(currentScores.get(windex) + undefined))/ denominator;
      else
        return 1./denominator;
    }

    public double logP(int windex) {
      if (windex < currentScores.dim())
        return currentScores.get(windex) + undefined - log(denominator);
      else
        return -log(denominator);
    }

    public double logOneMinusP(int windex) {
      if (windex < currentScores.dim())
        return log(denominator - exp(currentScores.get(windex) + undefined)) - log(denominator);
      else
        return log(denominator - 1) - log(denominator);
    }

    public void updatePositive(int windex, double alpha) {
      final VecIterator it = currentScores.nonZeroes();
      final double undefinedComponentIncrement = -exp(undefined)/denominator * alpha;
      double newDenominator = 1;
      int nzCount = 0;
      while (it.advance()) {
        double value = it.value() + undefined;
        if (it.index() == windex)
          value += (denominator - exp(value))/ denominator * alpha;
        else
          value += -exp(value)/denominator * alpha;

        nzCount++;
        newDenominator += exp(value);
        it.setValue(value - undefined - undefinedComponentIncrement);
      }
      undefined += undefinedComponentIncrement;
      newDenominator += exp(undefined) * (currentScores.dim() - nzCount);
      denominator = newDenominator;
    }

    public void updateNegative(int windex, double alpha) {
      final VecIterator it = currentScores.nonZeroes();
      double denomMinusCorrect = denominator - exp(currentScores.get(windex) + undefined);
      final double undefinedComponentIncrement = ((denominator - exp(undefined))/denominator - (denomMinusCorrect - exp(undefined))/denomMinusCorrect) * alpha;
      double newDenominator = 1;
      int nzCount = 0;
      while (it.advance()) {
        double value = it.value() + undefined;
        if (it.index() == windex)
          value += -exp(value)/denominator * alpha;
        else
          value += ((denominator - exp(value))/denominator - (denomMinusCorrect - exp(value))/denomMinusCorrect) * alpha;

        nzCount++;
        newDenominator += exp(value);
        it.setValue(value - undefined - undefinedComponentIncrement);
      }
      undefined += undefinedComponentIncrement;
      newDenominator += exp(undefined) * (currentScores.dim() - nzCount);
      denominator = newDenominator;
    }

    public void print(String[] words) {
      final ObjectMapper mapper = new ObjectMapper();
      final ObjectNode output = mapper.createObjectNode();
      output.put("undefined", undefined);
      output.put("denominator", denominator);
      final ObjectNode wordsNode = output.putObject("words");
      final int[] myIndices = new int[currentScores.size() + 1];
      final double[] myProbabs = new double[currentScores.size() + 1];
      final VecIterator nz = currentScores.nonZeroes();
      int index = 0;
      while (nz.advance()) {
        myIndices[index] = nz.index();
        myProbabs[index] = logP(nz.index());
        index++;
      }

      ArrayTools.parallelSort(myProbabs, myIndices);
      for (int i = 0; i < myIndices.length; i++) {
        final int windex= myIndices[i];
        wordsNode.put(words[windex], myProbabs[i]);
      }


      final ObjectWriter writer = mapper.writerWithDefaultPrettyPrinter();
      try {
        writer.writeValue(System.out, output);
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    }
  }

  private static final int STEPS = 10000;

  private static WordGenProbabilityProvider[] gradientDescentEMCascadeModel(String[] words, Seq<IntSeq>[] session, double alpha) {
    final FastRandom rng = new FastRandom(0);
    final WordGenProbabilityProvider[] providers = new WordGenProbabilityProvider[words.length + 1];
    for (int i = 0; i < providers.length; i++) {
      providers[i] = new WordGenProbabilityProvider(words.length, i);
    }

    for (int t = 0; t < STEPS; t++) {
      int index = rng.nextInt(session.length);
      final Seq<IntSeq> next = session[index];
      for (int i = 1; i < next.length(); i++) {
        final IntSeq prev = next.at(i - 1);
        final IntSeq current = next.at(i);
        for (int j = 0; j < current.length(); j++) {
          final int windex = current.intAt(j);
          double logOneMinusProb = providers[words.length].logOneMinusP(windex);
          for (int k = 0; k < prev.length(); k++) {
            logOneMinusProb += providers[prev.intAt(k)].logOneMinusP(windex);
          }

          int best = words.length; // word out of none
          double bestScore = providers[words.length].logP(windex) - providers[words.length].logOneMinusP(windex) + logOneMinusProb;

          for (int k = 0; k < prev.length(); k++) {
            final int gwindex = prev.intAt(k);
            final double score = providers[gwindex].logP(windex) - providers[gwindex].logOneMinusP(windex) + logOneMinusProb;
            if (score >= bestScore) {
              best = gwindex;
            }
          }

          for (int k = 0; k < prev.length(); k++) {
            final int gwindex = prev.intAt(k);
            if (gwindex == best)
              providers[gwindex].updatePositive(windex, alpha);
            else
              providers[gwindex].updateNegative(windex, alpha);
          }
        }
      }
    }
    return providers;
  }
  volatile static int index = 0;
  volatile static int dictIndex = 0;

  public static void main(String[] args) throws IOException {
    if (args.length < 2)
      throw new IllegalArgumentException("Need at least two arguments: mode and file to work with");
    if ("-dict".equals(args[0])) {
      final DictExpansion<CharSeq> expansion = new DictExpansion<>(500000, System.out);
      final String inputFileName = args[1];
      final ThreadPoolExecutor executor = ThreadTools.createBGExecutor("Creating DictExpansion", 100000);
      for (int i = 0; i < 100; i++) {
        CharSeqTools.processLines(new InputStreamReader(new GZIPInputStream(new FileInputStream(inputFileName))), new Action<CharSequence>() {
          CharSequence currentUser;
          Holder<Dictionary<CharSeq>> dumped = new Holder<>();
          long ts = 0;
          CharSequence query;

          @Override
          public void invoke(CharSequence line) {
            final CharSequence[] parts = new CharSequence[3];
            if (CharSeqTools.split(line, '\t', parts).length != 3)
              throw new IllegalArgumentException("Each input line must contain <uid>\\t<ts>\\t<query> triplet. This one: [" + line + "]@" + inputFileName + ":" + index + " does not.");
            final long ts = CharSeqTools.parseLong(parts[1]);
            if (parts[2].equals(query)) {
              this.ts = ts;
              return;
            }
            query = parts[2].toString();
//          if (!CharSeqTools.equals(parts[0], currentUser))
            final CharSeq[] words = new CharSeq[100];
            final int wcount = CharSeqTools.trySplit(new CharSeqAdapter(query), ' ', words);
            final Runnable item = () -> {
              expansion.accept(new ArraySeq<>(words, 0, wcount));
              if ((++index) % 1000000 == 0 && dumped.getValue() != expansion.result()) {
                try {
                  dumped.setValue(expansion.result());
                  System.out.println("Dump dictionary #" + dictIndex);
                  expansion.print(new FileWriter(StreamTools.stripExtension(inputFileName) + "-big-" + dictIndex + ".dict"));
                  dictIndex++;
                } catch (Exception e) {
                  e.printStackTrace();
                }
              }
            };
            final BlockingQueue<Runnable> queue = executor.getQueue();
            if (queue.remainingCapacity() == 0) {
              try {
                queue.put(item);
              } catch (InterruptedException e) {
                throw new RuntimeException(e);
              }
            } else executor.execute(item);
//          expansion.
          }
        });
        expansion.print(new FileWriter(StreamTools.stripExtension(inputFileName) + "-big.dict"));
      }
    }
  }
}
