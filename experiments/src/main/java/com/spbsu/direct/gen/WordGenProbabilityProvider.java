package com.spbsu.direct.gen;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectWriter;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.spbsu.commons.io.codec.seq.Dictionary;
import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.math.vectors.VecIterator;
import com.spbsu.commons.math.vectors.impl.vectors.SparseVec;
import com.spbsu.commons.seq.CharSeq;
import com.spbsu.commons.seq.IntSeq;
import com.spbsu.commons.util.ArrayTools;
import gnu.trove.map.TIntIntMap;
import gnu.trove.map.hash.TIntIntHashMap;

import java.io.IOException;
import java.io.Writer;

import static java.lang.Math.*;

/**
 * User: solar
 * Date: 12.11.15
 * Time: 11:33
 */
public class WordGenProbabilityProvider {
  public static final int MINIMUM_STATISTICS_TO_OUTPUT = 20;
  private final int aindex;
  SparseVec beta;
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

  final ThreadLocal<ObjectMapper> mapper = new ThreadLocal<ObjectMapper>() {
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
}
