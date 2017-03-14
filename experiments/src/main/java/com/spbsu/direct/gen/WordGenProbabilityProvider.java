package com.spbsu.direct.gen;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.ObjectWriter;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.spbsu.commons.io.codec.seq.Dictionary;
import com.spbsu.commons.math.AnalyticFunc;
import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.math.vectors.VecIterator;
import com.spbsu.commons.math.vectors.impl.vectors.SparseVec;
import com.spbsu.commons.seq.*;
import com.spbsu.commons.util.ArrayTools;
import gnu.trove.map.TIntIntMap;
import gnu.trove.map.hash.TIntIntHashMap;
import gnu.trove.procedure.TIntDoubleProcedure;

import java.io.IOException;
import java.io.Writer;
import java.util.Iterator;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static java.lang.Math.*;

/**
 * User: solar
 * Date: 12.11.15
 * Time: 11:33
 */
public class WordGenProbabilityProvider {
  public static final int MINIMUM_STATISTICS_TO_OUTPUT = 20;
  final int aindex;
  SparseVec beta;
  private double poissonLambdaSum = 1;
  private double poissonLambdaCount = 1;
  private double denominator;
  private double undefined;
  private static Pattern headerPattern = Pattern.compile("(\\[(?:[^, ]+(?:, )?)*\\]): \\{(.*)");

  private double dpAlpha = Double.NaN;

  public WordGenProbabilityProvider(int dim, int windex) {
    aindex = windex;
    beta = new SparseVec(dim, 0);
    denominator = dim; // 1 + \sum_1^dim e^0
  }

  public WordGenProbabilityProvider(CharSequence presentation, Dictionary<CharSeq> dict) {
    final String json;
    {
      final SeqBuilder<CharSeq> phraseBuilder = new ArraySeqBuilder<>(CharSeq.class);
      final Matcher matcher = headerPattern.matcher(presentation);
      if (!matcher.find())
        throw new IllegalArgumentException(presentation.toString());

      final String phrase = matcher.group(1);
      final CharSequence[] parts = CharSeqTools.split(phrase.subSequence(1, phrase.length() - 1), ", ");
      for (final CharSequence part : parts) {
        phraseBuilder.add(CharSeq.create(part.toString()));
      }
      json = "{" + matcher.group(2) + "}";
      aindex = dict.search(phraseBuilder.build());
    }
    final ObjectReader reader = mapper.get().reader();
    beta = new SparseVec(dict.size());

    try {
      final JsonNode node = reader.readTree(json);
      poissonLambdaSum = node.get("poissonSum").asDouble();
      poissonLambdaCount = node.get("poissonCount").asDouble();
      denominator = node.get("denominator").asDouble();
      undefined = node.get("undefined").asDouble();
      final JsonNode words = node.get("words");
      final Iterator<Map.Entry<String, JsonNode>> fieldsIt = words.fields();
      final SeqBuilder<CharSeq> phraseBuilder = new ArraySeqBuilder<>(CharSeq.class);
      double totalWeight = 0;
      while (fieldsIt.hasNext()) {
        final Map.Entry<String, JsonNode> next = fieldsIt.next();
        final String phrase = next.getKey();
        final CharSequence[] parts = CharSeqTools.split(phrase.subSequence(1, phrase.length() - 1), ", ");
        for (final CharSequence part : parts) {
          phraseBuilder.add(CharSeq.create(part.toString()));
        }

        final double weight = log(denominator * next.getValue().asDouble());
        totalWeight += weight;
        beta.set(dict.search(phraseBuilder.build()), weight);
        phraseBuilder.clear();

      }

      { // DP schema
        dpAlpha = optimalExpansionDP(totalWeight, beta.size());
      }
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  public double logP(int variant, IntSeq gen) {
    double result = MathTools.logPoissonProbability(poissonLambdaSum / poissonLambdaCount, Integer.bitCount(variant));
    for (int i = 0; i < gen.length(); i++, variant >>= 1) {
      final boolean positive = (variant & 1) == 1;
      final int windex = gen.intAt(i);

      result += positive ? log(pAB(windex)) : 0;//log(1 - pAB(windex));
    }
    return result;
  }

  private final TIntIntMap weightsPools = new TIntIntHashMap();
  private int poolSize = 0;

  public void update(int variant, IntSeq gen, double alpha, Dictionary<CharSeq> dict, boolean debug) {
    final int length = gen.length();
    poissonLambdaCount++;
    poissonLambdaSum += Integer.bitCount(variant);
    if (debug)
      System.out.print(wordText(aindex, dict) + "->");
    for (int i = 0; i < length; i++, variant >>= 1) {
      final boolean positive = (variant & 1) == 1;
      final int windex = gen.intAt(i);
      if (positive && debug) {
        System.out.print(" " + wordText(windex, dict));
      }
      if (positive)
        weightsPools.adjustOrPutValue(windex, +1, +1);
    }
    if (debug)
      System.out.println();
    poolSize++;
    if (!debug && poolSize < poissonLambdaCount / 20.)
      return;
    update(weightsPools, alpha);
    weightsPools.clear();
    poolSize = 0;
  }

  private Dictionary<CharSeq> dict;

  private String wordText(int index, Dictionary<CharSeq> dict) {
    if (index < 0 || index >= dict.size())
      return SimpleGenerativeModel.EMPTY_ID;
    this.dict = dict;
    return dict.get(index).toString();
  }

  public void update(TIntIntMap weightsPools, double alpha) {
    alpha /= poolSize;

    final int[] updates = weightsPools.keys();
    final int[] count = weightsPools.values();
    { // gradient step
      double gradientTermSum = 0;
      double[] oldValues = new double[updates.length];
      for (int i = 0; i < updates.length; i++) {
        final int windex = updates[i];
        double value = oldValues[i] = beta.get(windex);
        double pAB = exp(value + undefined) / denominator;
        if (abs(pAB) < 1e-10)
          continue;
        gradientTermSum += count[i];
      }
      if (gradientTermSum == 0)
        return;
      final double newUndefined = undefined - alpha * gradientTermSum * pGen(-1);

      final VecIterator it = beta.nonZeroes();
      int nzCount = 0;
      double newDenominator = 1;
      while (it.advance()) { // updating all non zeroes as if they were negative, then we will change the gradient for positives
        double value = it.value() + undefined;
        final double pAI = exp(value) / denominator;
        value += -alpha * gradientTermSum * pAI;
        if (abs(value) > 100)
          System.out.println();
        nzCount++;
        final double exp = exp(value);
        newDenominator += exp;
        it.setValue(value - newUndefined);
      }

      for (int i = 0; i < updates.length; i++) {
        final int windex = updates[i];
        double value = oldValues[i] + undefined;
        final double pAB = exp(value) / denominator;

        if (value != undefined) { // reverting changes made in previous loop for this example
          final double exp = exp(value - alpha * gradientTermSum * pAB);
          newDenominator -= exp;
          if (newDenominator < 0)
            System.out.println();
          nzCount--;
        }
        final double positiveGradientTerm = 1;
        final double grad = -(gradientTermSum - count[i] * positiveGradientTerm) * pAB + count[i] * positiveGradientTerm * (1 - pAB);
        value += alpha * grad;
        if (abs(value) > 100)
          System.out.println();

        nzCount++;
        newDenominator += exp(value);
        beta.set(windex, value - newUndefined);
      }
      newDenominator += exp(undefined) * (beta.dim() - nzCount);

      undefined = newUndefined;
      if (newDenominator < 0)
        System.out.println();
      denominator = newDenominator;
    }
  }

  WordGenProbabilityProvider[] all;
  double probab;

  private double pAB(int windex) {
    //      final double pB = all[windex].probab;
//      return pB + (1 - pB) * pGenB;
    return pGen(windex);
  }

  public double pGen(int windex) {
    final double logPGenB;
    if (!Double.isNaN(dpAlpha)) { // DP model
      final double statPower = denominator - beta.dim();
      if (windex < 0 || windex >= beta.dim() || beta.get(windex) < MathTools.EPSILON)
        // log(\frac{1}{N - m + 1} {\alpha \over \alpha + n - 1}),
        // where N -- vocabulary size, m -- number of words, occurred in statistics, \alpha -- DP expansion parameter, n -- statistics power
        logPGenB = -log(beta.dim() - beta.size() + 1) + log(dpAlpha) - log(dpAlpha + statPower - 1);
      else
        // log({\e^s_i \over \alpha + n - 1}),
        // where N -- vocabulary size, m -- number of words, occurred in statistics, \alpha -- DP expansion parameter, n -- statistics power
        logPGenB = log(beta.get(windex) - 1) - log(dpAlpha + statPower - 1);
    }
    else {
      if (windex < 0)
        logPGenB = undefined - log(denominator);
      else if (windex < beta.dim())
        logPGenB = beta.get(windex) + undefined - log(denominator);
      else
        logPGenB = -log(denominator);
    }
    return exp(logPGenB);
  }

  private static final ThreadLocal<ObjectMapper> mapper = new ThreadLocal<ObjectMapper>() {
    @Override
    protected ObjectMapper initialValue() {
      return new ObjectMapper();
    }
  };

  public void print(Dictionary<CharSeq> words, Writer to, boolean limit) {
    if (poissonLambdaCount < MINIMUM_STATISTICS_TO_OUTPUT && limit)
      return;
    final ObjectNode output = mapper.get().createObjectNode();
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
    int wordsCount = 0;
    for (int i = myIndices.length - 1; i >= 0; i--) {
        if(myProbabs[i] < 1e-6 && limit)
          break;
      final int windex = myIndices[i];
      wordsCount++;
      if (words.size() > windex)
        wordsNode.put(words.get(windex).toString(), myProbabs[i]);
      else
        wordsNode.put(SimpleGenerativeModel.EMPTY_ID, myProbabs[i]);
    }

    if (wordsCount < 2)
      return;

    final ObjectWriter writer = mapper.get().writerWithDefaultPrettyPrinter();
    try {
      if (aindex < dict.size())
        to.append(dict.get(aindex).toString()).append(": ");
      else
        to.append(SimpleGenerativeModel.EMPTY_ID).append(": ");

      to.append(writer.writeValueAsString(output));
      to.append("\n");
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  public void init(WordGenProbabilityProvider[] providers, Dictionary<CharSeq> dict) {
    all = providers;
    this.dict = dict;
    final VecIterator nz = beta.nonZeroes();
    int nzCount = 0;
    denominator = 1;
    while (nz.advance()) {
      final int count = (int) nz.value();
      nz.setValue(log(count));
      denominator += exp(nz.value());
      nzCount++;
    }
    denominator += beta.dim() - nzCount;
  }

  public void visitVariants(final TIntDoubleProcedure todo) {
    final VecIterator it = beta.nonZeroes();
    while (it.advance()) {
      todo.execute(it.index(), pGen(it.index()));
    }
  }

  double best = Double.NEGATIVE_INFINITY;
  public boolean isMeaningful(int index) {
    return pGen(index) > 0.0005;
  }

  /**
   * m = \alpha log(1 + \frac{n}{\alpha})
   * where n -- draws count, m -- classes found
   */
  private double optimalExpansionDP(double statPower, int classes) {
    return MathTools.bisection(0, classes, new AnalyticFunc.Stub() {
      @Override
      public double value(double x) {
        return x * log(1 + statPower / x) - classes;
      }
    });
  }
}
