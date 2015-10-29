package com.spbsu.direct;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectWriter;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.spbsu.commons.func.Action;
import com.spbsu.commons.io.StreamTools;
import com.spbsu.commons.io.codec.seq.DictExpansion;
import com.spbsu.commons.io.codec.seq.Dictionary;
import com.spbsu.commons.io.codec.seq.ListDictionary;
import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.math.io.Vec2CharSequenceConverter;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecIterator;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.vectors.SparseVec;
import com.spbsu.commons.math.vectors.impl.vectors.VecBuilder;
import com.spbsu.commons.seq.*;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.commons.util.Holder;
import com.spbsu.commons.util.ThreadTools;
import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.map.TIntIntMap;
import gnu.trove.map.hash.TIntIntHashMap;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.zip.GZIPInputStream;

import static com.spbsu.commons.math.vectors.VecTools.l1;
import static java.lang.Math.*;

/**
 * User: solar
 * Date: 07.10.15
 * Time: 15:56
 */
public class BroadMatch {
  static boolean debug = true;

  public static final String EMPTY_ID = "##EMPTY##";

  public static class WordGenProbabilityProvider {
    private final int aindex;
    private SparseVec beta;
    private double poissonLambdaSum = 1;
    private double poissonLambdaCount = 1;
    private double denominator;
    private double undefined;

    public WordGenProbabilityProvider(int dim, int windex) {
      aindex = windex;
      beta = new SparseVec(dim, 0);
      denominator = dim; // 1 + \sum_1^dim e^0
    }

    public double logP(int variant, IntSeq gen) {
      double result = MathTools.logPoissonProbability(poissonLambdaSum/poissonLambdaCount, Integer.bitCount(variant));
      for (int i = 0; i < gen.length(); i++, variant >>= 1) {
        if ((variant & 1) != 0)
          result += log(pAB(gen.intAt(i)));
      }
      return result;
    }

    public void update(int variant, IntSeq gen, double alpha, ListDictionary<CharSeq> dict, boolean debug) {
      final int length = gen.length();
      if (debug)
        System.out.print(wordText(aindex, dict) + "->");
      for (int i = 0; i < length; i++, variant >>= 1) {
        if ((variant & 1) == 0)
          continue;
        pool.add(gen.intAt(i));
        if (debug)
          System.out.print(" " + wordText(gen.intAt(i), dict));
      }
      if (debug)
        System.out.println();
      poissonLambdaCount++;
      poissonLambdaSum += Integer.bitCount(length);
      if (pool.length() < beta.size() / 10)
        return;
      update(this.pool.build(), alpha);
      this.pool = new IntSeqBuilder();

    }

    private String wordText(int index, ListDictionary<CharSeq> dict) {
      if (index < 0 || index >= dict.size())
        return EMPTY_ID;
      return dict.get(index).toString();
    }

    private IntSeqBuilder pool = new IntSeqBuilder();
    public void update(IntSeq pool, double alpha) {
      final TIntIntMap uniq = new TIntIntHashMap(pool.length());
      for(int i = 0; i < pool.length(); i++) {
        uniq.adjustOrPutValue(pool.at(i), 1, 1);
      }

      /*
          G = p(a->b) {(1 - p(b)) \over p(b) + p(a->b) (1 - p(b))}
          Gradient for positives: G (1 - p(a->i))
          Gradient for negatives: -G p(a->i)
          we sum up the G term across all links and then use this sum in update cycle
       */
      final int[] updates = uniq.keys();
      final int[] count = uniq.values();
      { // gradient step
        double gradientTermSum = 0;
        double[] oldValues = new double[updates.length];
        for (int i = 0; i < updates.length; i++) {
          final int windex = updates[i];
          double pB = all[windex].probab;
          double value = oldValues[i] = beta.get(windex);
          double pGenB = exp(value) / denominator;
          if (abs(pGenB) < 1e-10)
            continue;
          gradientTermSum += count[i] * (1 - pB) / (pB + pGenB * (1 - pB)) * pGenB;
        }
        if (gradientTermSum == 0)
          return;
        final double newUndefined = undefined - alpha * gradientTermSum * pGen(-1);

        final VecIterator it = beta.nonZeroes();
        int nzCount = 0;
        int newDenominator = 1;
        while (it.advance()) { // updating all non zeroes as if they were negative, then we will change the gradient for positives
          double value = it.value() + undefined;
          final double pAI = exp(value) / denominator;
          value += -alpha * gradientTermSum * pAI;
          nzCount++;
          newDenominator += exp(value);
          it.setValue(value - newUndefined);
        }

        for (int i = 0; i < updates.length; i++) {
          final int windex = updates[i];
          double value = oldValues[i] + undefined;
          final double pB = all[windex].probab;
          final double pAB = exp(value) / denominator;

          if (value != undefined) { // reverting changes made in previous loop for this example
            newDenominator -= exp(value - alpha * gradientTermSum * pAB);
            if (newDenominator < 0)
              System.out.println();
            nzCount--;
          }
          final double positiveGradientTerm = (1 - pB) / (pB + pAB * (1 - pB) + 1e-10) * pAB;
          final double grad = -(gradientTermSum - count[i] * positiveGradientTerm) * pAB + count[i] * positiveGradientTerm * (1 - pAB);
          if (grad > 100)
            System.out.println();
          value += alpha * grad;
          nzCount++;
          newDenominator += exp(value);
          beta.set(windex, value - newUndefined);
        }

        undefined = newUndefined;
        newDenominator += exp(undefined) * (beta.dim() - nzCount);
        if (newDenominator < 0)
          System.out.println();
        denominator = newDenominator;
      }
    }

    WordGenProbabilityProvider[] all;
    double probab;
    private double pAB(int windex) {
      final double pGenB = pGen(windex);
      final double pB = all[windex].probab;
      return pB + (1 - pB) * pGenB;
    }

    private double pGen(int windex) {
      final double logPGenB;
      if (windex < 0)
        logPGenB = undefined - log(denominator);
      else if (windex < beta.dim())
        logPGenB = beta.get(windex) + undefined - log(denominator);
      else
        logPGenB = -log(denominator);
      if (Double.isNaN(logPGenB)) {
        System.out.println();
      }
      return exp(logPGenB);
    }

    public void print(ListDictionary<CharSeq> words, Writer to) {
      final ObjectMapper mapper = new ObjectMapper();
      final ObjectNode output = mapper.createObjectNode();
      output.put("poissonSum", poissonLambdaSum);
      output.put("poissonCount", poissonLambdaCount);
      output.put("undefined", undefined);
      output.put("denominator", denominator);
      final ObjectNode wordsNode = output.putObject("words");
      final int[] myIndices = new int[beta.size() + 1];
      final double[] myProbabs = new double[beta.size() + 1];
      final VecIterator nz = beta.nonZeroes();
      int index = 0;
      while (nz.advance()) {
        myIndices[index] = nz.index();
        myProbabs[index] = pGen(nz.index());
        index++;
      }

      ArrayTools.parallelSort(myProbabs, myIndices);
      for (int i = myIndices.length - 1; i >= 0; i--) {
        if(myProbabs[i] < 1e-4)
          break;
        final int windex= myIndices[i];
        if(words.size() > windex)
          wordsNode.put(words.get(windex).toString(), myProbabs[i]);
        else
          wordsNode.put(EMPTY_ID, myProbabs[i]);
      }


      final ObjectWriter writer = mapper.writerWithDefaultPrettyPrinter();
      try {
        to.append(writer.writeValueAsString(output));
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    }

    public void init(WordGenProbabilityProvider[] providers, Vec freqs, double totalUnigramFreq, double totalBigramFreq) {
      all = providers;
      final double pA = probab;
      final double pairsFreq = l1(beta);
      final int totalKnown = beta.size();
      final double unexpectedGenProbab = 1./(pairsFreq + totalKnown)/(providers.length - beta.size());
      final VecIterator nz = beta.nonZeroes();
      int nzCount = 0;
      denominator = 1;
      while (nz.advance()) {
        final int windex = nz.index();
        final int count = (int)nz.value();
        final double pPair = (count + 1.) / (totalBigramFreq + all.length);
        final double pB = all[windex].probab;
        double pGenB = (pPair - pA * pB) / (1 - pB) / pA;
        if (pGenB < 0)
          pGenB = unexpectedGenProbab/2.;
        nz.setValue(log(pGenB) - log(unexpectedGenProbab));
        denominator += exp(nz.value());
        nzCount++;
      }
      denominator += beta.dim() - nzCount;
      if (denominator < 0)
        System.out.println();
    }
  }

  volatile static int index = 0;
  volatile static int dictIndex = 0;
  private final static Holder<Dictionary<CharSeq>> dumped = new Holder<>();

  public static void main(String[] args) throws IOException {
    if (args.length < 2)
      throw new IllegalArgumentException("Need at least two arguments: mode and file to work with");
    switch (args[0]) {
      case "-dict": {
        final DictExpansion<CharSeq> expansion = new DictExpansion<>(300000, System.out);
        final String inputFileName = args[1];
        final ThreadPoolExecutor executor = ThreadTools.createBGExecutor("Creating DictExpansion", 100000);
        for (int i = 0; i < 100; i++) {
          CharSeqTools.processLines(new InputStreamReader(new GZIPInputStream(new FileInputStream(inputFileName))), (Action<CharSequence>) line -> {
            final CharSequence[] parts = new CharSequence[3];
            if (CharSeqTools.split(line, '\t', parts).length != 3)
              throw new IllegalArgumentException("Each input line must contain <uid>\\t<ts>\\t<query> triplet. This one: [" + line + "]@" + inputFileName + ":" + index + " does not.");
//          if (!CharSeqTools.equals(parts[0], currentUser))
            final ArraySeq<CharSeq> seq = convertToSeq(parts[2]);
            final Runnable item = () -> {
              expansion.accept(seq);
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
          });
          expansion.print(new FileWriter(StreamTools.stripExtension(inputFileName) + "-big.dict"));
        }
        break;
      }
      case "-depends": {
        final double alpha = 0.2;
        final String inputFileName = args[2];
        final Vec freqs;

        final ListDictionary<CharSeq> dict;
        { // dict
          final VecBuilder freqBuilder = new VecBuilder();
          final List<Seq<CharSeq>> dictSeqs = new ArrayList<>();

          CharSeqTools.processLines(new FileReader(args[1]), (Action<CharSequence>) line -> {
            final CharSequence[] split = CharSeqTools.split(line, '\t', new CharSequence[2]);
            freqBuilder.append(CharSeqTools.parseDouble(split[1]));
            final CharSequence[] parts = CharSeqTools.split(split[0].subSequence(1, split[0].length() - 1), ", ");
            final SeqBuilder<CharSeq> builder = new ArraySeqBuilder<>(CharSeq.class);
            for (final CharSequence part : parts) {
              builder.add(new CharSeqAdapter(part.toString()));
            }
            dictSeqs.add(builder.build());
          });
          //noinspection unchecked
          dict = new ListDictionary<CharSeq>(dictSeqs.toArray(new Seq[dictSeqs.size()]));
          freqs = freqBuilder.build();
        }
        final double totalUnigramFreq = l1(freqs);

        final WordGenProbabilityProvider[] providers = new WordGenProbabilityProvider[dict.size() + 1];
        { // stats
          for (int i = 0; i < providers.length; i++) {
            providers[i] = new WordGenProbabilityProvider(dict.size(), i);
          }
          final String statsFile = StreamTools.stripExtension(inputFileName) + ".stats.gz";
          final Vec2CharSequenceConverter converter = new Vec2CharSequenceConverter();
          CharSeqTools.processLines(new InputStreamReader(new GZIPInputStream(new FileInputStream(statsFile))), new Action<CharSequence>() {
            @Override
            public void invoke(CharSequence sequence) {
              final CharSequence[] split = CharSeqTools.split(sequence, '\t');

              final WordGenProbabilityProvider provider;
              if (!split[0].equals(EMPTY_ID)) {
                final CharSequence[] parts = CharSeqTools.split(split[0].subSequence(1, split[0].length() - 1), ", ");
                final SeqBuilder<CharSeq> builder = new ArraySeqBuilder<>(CharSeq.class);
                for (final CharSequence part : parts) {
                  builder.add(new CharSeqAdapter(part.toString()));
                }

                final int index = dict.parse(builder.build()).intAt(0);
                if (index < 0)
                  return;
                provider = providers[index];
              }
              else provider = providers[providers.length - 1];
              final Vec vec = converter.convertFrom(split[1]);
              provider.beta = VecTools.copySparse(vec); // optimize storage space
            }
          });
          double totalBigramFreq = 0;
          for(int i = 0; i < providers.length; i++) {
            totalBigramFreq += l1(providers[i].beta);
          }

          for(int i = 0; i < providers.length; i++) {
            providers[i].probab = l1(providers[i].beta) / totalBigramFreq;
          }
          for(int i = 0; i < providers.length; i++) {
            providers[i].init(providers, freqs, totalUnigramFreq, totalBigramFreq);
          }
        }
        CharSeqTools.processLines(new InputStreamReader(new GZIPInputStream(new FileInputStream(inputFileName))), new Action<CharSequence>() {
          int index = 0;
          long ts;
          String query;
          String user;

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
            final String prev = parts[0].equals(this.user) && ts - this.ts < TimeUnit.MINUTES.toSeconds(30) ? this.query : null;
            this.query = parts[2].toString();
            this.user = parts[0].toString();
            this.ts = ts;
            if (prev != null) {
              processGeneration(prev, alpha);
            }
            if (++index % 1000000 == 0) {
              try (final Writer out = new OutputStreamWriter(new FileOutputStream("output-" + (index / 1000000) + ".txt"))) {
                for (int i = 0; i < providers.length; i++) {
                  final WordGenProbabilityProvider provider = providers[i];
                  if (provider.poissonLambdaCount < 20)
                    continue;
                  if (i < dict.size())
                    out.append(dict.get(i).toString()).append(": ");
                  else
                    out.append(EMPTY_ID).append(": ");
                  provider.print(dict, out);
                  out.append("\n");
                }
              } catch (IOException e) {
                e.printStackTrace();
              }
            }
          }

          private final TDoubleArrayList window = new TDoubleArrayList(1000);
          private double windowSum = 0;
          private void processGeneration(String prev, double alpha) {
            final IntSeq prevQSeq = dropUnknown(dict.parse(convertToSeq(prev)));
            final IntSeq currentQSeq = dropUnknown(dict.parse(convertToSeq(query)));
            final WordGenProbabilityProvider zeroElementProvider = providers[providers.length - 1];

            if (prevQSeq.length() * currentQSeq.length() > 10) // too many variants of bipartite graph
              return;
            final int variantsCount = 1 << (prevQSeq.length() * currentQSeq.length());
            final int mask = (1 << currentQSeq.length()) - 1;
            int bestVariant = 0;
            double bestLogProBab = Double.NEGATIVE_INFINITY;
            { // expectation
              for (int p = 0; p < variantsCount; p++) {
                double variantLogProBab = 0;
                {
                  int variant = p;
                  int generated = 0;
                  for (int i = 0; i < prevQSeq.length(); i++, variant >>= currentQSeq.length()) {
                    final int fragment = variant & mask;
                    generated |= fragment;
                    final int index = prevQSeq.intAt(i);
                    if (index < 0)
                      continue;
                    variantLogProBab += providers[index].logP(fragment, currentQSeq);
                  }
                  variantLogProBab += zeroElementProvider.logP((~generated & mask), currentQSeq);
                }
                if (variantLogProBab > bestLogProBab) {
                  bestLogProBab = variantLogProBab;
                  bestVariant = p;
                }
              }
            }
            if (!Double.isFinite(bestLogProBab))
              return;
            if (Double.isNaN(bestLogProBab))
              System.out.println();
            { // maximization gradient descent step

              int generated = 0;
              windowSum += bestLogProBab;
              window.add(bestLogProBab);
              final double remove;
              if (window.size() > 1000) {
                remove = window.removeAt(0);
                if (Double.isNaN(bestLogProBab))
                  windowSum -= remove;
              }
              if (Double.isNaN(windowSum)) {
                System.out.println();
              }

              boolean debug = BroadMatch.debug && (index % 1000 == 0);
              if (debug)
                System.out.println(prev + " -> " + query + " " + windowSum / window.size());
              for (int i = 0; i < prevQSeq.length(); i++, bestVariant >>= currentQSeq.length()) {
                final int fragment = bestVariant & mask;
                generated |= fragment;
                final int windex = prevQSeq.intAt(i);
                if (windex < 0)
                  continue;
                providers[windex].update(fragment, currentQSeq, alpha, dict, debug);
              }
              zeroElementProvider.update(~generated & mask, currentQSeq, alpha, dict, debug);
            }
          }
        });
        break;
      }
      case "-stats": {
        final Vec2CharSequenceConverter converter = new Vec2CharSequenceConverter();
        final ListDictionary<CharSeq> dict;
        {
          final List<Seq<CharSeq>> dictSeqs = new ArrayList<>();

          CharSeqTools.processLines(new FileReader(args[1]), (Action<CharSequence>) line -> {
            final CharSequence[] split = CharSeqTools.split(line, '\t', new CharSequence[2]);
            final CharSequence[] parts = CharSeqTools.split(split[0].subSequence(1, split[0].length() - 1), ", ");
            final SeqBuilder<CharSeq> builder = new ArraySeqBuilder<>(CharSeq.class);
            for (final CharSequence part : parts) {
              builder.add(new CharSeqAdapter(part.toString()));
            }
            dictSeqs.add(builder.build());
          });
          //noinspection unchecked
          dict = new ListDictionary<CharSeq>(dictSeqs.toArray((new Seq[dictSeqs.size()])));
        }
        final String inputFileName = args[2];
        final SparseVec[] stats = new SparseVec[dict.size() + 1];

        for (int i = 0; i < stats.length; i++) {
          stats[i] = new SparseVec(dict.size());
        }
        CharSeqTools.processLines(new InputStreamReader(new GZIPInputStream(new FileInputStream(inputFileName))), new Action<CharSequence>() {
          int index = 0;
          long ts;
          String query;
          String user;

          @Override
          public void invoke(CharSequence line) {
            final CharSequence[] parts = new CharSequence[3];
            if (CharSeqTools.split(line, '\t', parts).length != 3)
              throw new IllegalArgumentException("Each input line must contain <uid>\\t<ts>\\t<query> triplet. This one: [" + line + "]@" + inputFileName + ":" + index + " does not.");
            final long ts = CharSeqTools.parseLong(parts[1]);
            final CharSequence query = parts[2];
            if (query.equals(this.query)) {
              this.ts = ts;
              return;
            }
            final CharSequence uid = parts[0];
            if (!uid.equals(this.user)) {
              if (this.query != null) { // session end
                final IntSeq prevQSeq = dropUnknown(dict.parse(convertToSeq(this.query)));
                for (int i = 0; i < prevQSeq.length(); i++) {
                  stats[prevQSeq.intAt(i)].adjust(dict.size(), 1.);
                }
              }
              { // session start
                final IntSeq currentQSeq = dropUnknown(dict.parse(convertToSeq(query)));
                for (int i = 0; i < currentQSeq.length(); i++) {
                  stats[dict.size()].adjust(currentQSeq.intAt(i), 1.);
                }
              }
            }
            final String prev = uid.equals(this.user) && ts - this.ts < TimeUnit.MINUTES.toSeconds(30) ? this.query : null;
            this.query = query.toString();
            this.user = uid.toString();
            this.ts = ts;
            if (prev != null) {
              final IntSeq prevQSeq = dropUnknown(dict.parse(convertToSeq(prev)));
              final IntSeq currentQSeq = dropUnknown(dict.parse(convertToSeq(this.query)));
              for (int i = 0; i < prevQSeq.length(); i++) {
                for (int j = 0; j < currentQSeq.length(); j++) {
                  stats[prevQSeq.intAt(i)].adjust(currentQSeq.intAt(j), 1.);
                }
              }
            }
            if (++index % 10000000 == 0) {
              final String outputFile = StreamTools.stripExtension(inputFileName) + "-" + (index / 10000000) + ".stats";
              System.out.println("Dump " + outputFile);
              try (final Writer out = new OutputStreamWriter(new FileOutputStream(outputFile))) {
                for (int i = 0; i < stats.length; i++) {
                  final SparseVec stat = stats[i];
                  if (i < dict.size())
                    out.append(dict.get(i).toString());
                  else
                    out.append(EMPTY_ID);
                  out.append("\t");
                  out.append(converter.convertTo(stat));
                  out.append("\n");
                }
              } catch (IOException e) {
                e.printStackTrace();
              }
            }
          }

        });
        break;
      }
    }
  }

  private static IntSeq dropUnknown(IntSeq parse) {
    final IntSeqBuilder builder = new IntSeqBuilder();
    for (int i = 0; i < parse.length(); i++) {
      final int val = parse.intAt(i);
      if (val >= 0)
        builder.add(val);
    }
    return builder.build();
  }

  private static ArraySeq<CharSeq> convertToSeq(CharSequence word) {
    final CharSeq[] words = new CharSeq[100];
    final String query = word.toString();
    final int wcount = CharSeqTools.trySplit(new CharSeqAdapter(query), ' ', words);
    return new ArraySeq<>(words, 0, wcount);
  }
}
