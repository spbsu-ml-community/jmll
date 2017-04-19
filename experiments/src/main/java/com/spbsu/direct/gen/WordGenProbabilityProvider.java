package com.spbsu.direct.gen;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.ObjectWriter;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.spbsu.commons.io.codec.seq.Dictionary;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.math.AnalyticFunc;
import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.math.vectors.VecIterator;
import com.spbsu.commons.math.vectors.impl.vectors.SparseVec;
import com.spbsu.commons.seq.*;
import com.spbsu.commons.util.ArrayTools;
import gnu.trove.iterator.TIntIntIterator;
import gnu.trove.map.TIntIntMap;
import gnu.trove.map.hash.TIntIntHashMap;
import gnu.trove.procedure.TIntDoubleProcedure;

import java.io.IOException;
import java.io.Writer;
import java.lang.reflect.Array;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static java.lang.Math.*;

/**
 * User: solar
 * Date: 12.11.15
 * Time: 11:33
 */
public class WordGenProbabilityProvider {
  public static boolean DEBUG = true;

  public static final int MINIMUM_STATISTICS_TO_OUTPUT = 20;

  private static final int POOL_SIZE = 20;
  private static final int BUFFER_SIZE = 10;
  private static final int MAX_ITERATION_COUNT = 1000;
  private static final double EPS = 1e-5;
  private static final double INF = 1e18;

  private static Pattern headerPattern = Pattern.compile("(\\[(?:[^, ]+(?:, )?)*\\]): \\{(.*)");

  private Dictionary<CharSeq> dict;

  private double poissonLambdaSum = 1;
  private double poissonLambdaCount = 1;

  private double dpAlpha = 1.0; // TODO: think about good initialization
  private double undefinedGamma = 0;
  private double denominator;

  private SparseVec count;
  private SparseVec gamma;

  private int totalCount;
  private int uniqueTermsCount;
  private int newTermsTotalCount;
  private final TIntIntMap newTermsCount = new TIntIntHashMap();
  private final ArrayList<Integer> generationIndices = new ArrayList<>();

  final int providerIndex; // for debug purposes

  public WordGenProbabilityProvider(final int index,
                                    final Dictionary<CharSeq> dictionary) {
    dict = dictionary;
    providerIndex = index;
    count = new SparseVec(dictionary.size(), 0); // without empty word
    gamma = new SparseVec(dictionary.size(), 0); // parameters to learn

    denominator = 1;
  }


  /**
   * Returns the log-probability of fragment generation by the current term
   * p = Poisson(\lambda) * p(fragment | current term)
   * <p>
   * If the given fragment is empty, p(fragment | current term) = \frac{1}{1 + \sum_i e^gamma[i]} * \frac{n - 1}{\alpha + n - 1}
   * Otherwise, p(fragment | current term) = \prod_{t \in fragment} p(t | current term)
   * <p>
   * If t is new class, p(t | current term) = \frac{1}{N - m} * \frac{\alpha}{\alpha + n - 1}
   * Otherwise, p(t | current term) = \frac{e^gamma[t]}{1 + \sum_i e^gamma[i]} * \frac{n - 1}{\alpha + n - 1}
   * <p>
   * Where:
   * n -- the number of the current word
   * N -- total count of classes (size of the dictionary)
   * m -- count of classes for the current term
   * <p>
   * Additional description:
   * denominator = 1 + \sum_i e^gamma[i]
   *
   * @param variant  the mask of generated terms
   * @param fragment the sequence of terms
   * @return log-probability of fragment generation by the current term
   */
  public double logP(final int variant,
                     final IntSeq fragment) {
    int n = totalCount;
    int uniqueCount = uniqueTermsCount;
    double denominator = this.denominator;

    double result = MathTools.logPoissonProbability(poissonLambdaSum / poissonLambdaCount, Integer.bitCount(variant));

    int [] newTerms = new int[BUFFER_SIZE];
    int tail = 0;
    ArrayTools.fill(newTerms, -1);

    if (variant == 0) {
      result += -log(denominator) + log(n) - log(dpAlpha + n);
    } else {
      for (int i = 0, mask = variant; i < fragment.length(); i++, mask >>= 1) {
        if ((mask & 1) == 0) {
          continue; // skip not generated term
        }

        ++n; // met new generated word
        final int index = fragment.intAt(i);

        if (count.get(index) == 0) {
          if (ArrayTools.indexOf(index, newTerms) == -1) { // TODO: optimize performance
            newTerms[tail++] = index; // mark that we have met this term
            denominator += exp(undefinedGamma);

            result += -log(dict.size() - uniqueCount) + log(dpAlpha) - log(dpAlpha + n - 1);
            ++uniqueCount;
          } else {
            result += undefinedGamma - log(denominator) + log(n - 1) - log(dpAlpha + n - 1);
          }
        } else {
          result += gamma.get(index) - log(denominator) + log(n - 1) - log(dpAlpha + n - 1);
        }
      }
    }

    return result;
  }

  public void update(int variant,
                     final IntSeq seq,
                     final double alpha) {
    final int length = seq.length();

    poissonLambdaCount++;
    poissonLambdaSum += Integer.bitCount(variant);

    for (int i = 0; i < length; ++i, variant >>= 1) {
      if ((variant & 1) == 0) {
        continue; // skip not generated term
      }

      final int index = seq.intAt(i);

      if (count.get(index) == 0) {
        applyUpdates(alpha); // new term -- need to apply previous changes

        gamma.set(index, undefinedGamma);
        denominator += exp(undefinedGamma);

        ++uniqueTermsCount;
        count.adjust(index, 1);

        generationIndices.add(totalCount++);

        dpAlpha = findDpAlpha();
      } else {
        ++newTermsTotalCount;
        newTermsCount.adjustOrPutValue(index, 1, 1);
      }
    }

    // TODO: review
    if (!DEBUG && newTermsTotalCount / poissonLambdaCount < POOL_SIZE) {
      return;
    }

    applyUpdates(alpha);
  }

  private void gradientDescentGamma(double alpha) {
    alpha /= totalCount; // TODO: is it okay?

    double delta = INF;

    for (int iteration = 1; iteration < MAX_ITERATION_COUNT && delta > EPS; ++iteration) {
      double newDenominator = 1;

      VecIterator it = count.nonZeroes();
      while (it.advance()) {
        int index = it.index();

        double x = gamma.get(index);
        double gradient = (count.get(index) - 1) - (totalCount - uniqueTermsCount) * exp(x) / denominator;
        double new_x = x + alpha * gradient;

        gamma.set(index, new_x);
        newDenominator += exp(new_x);

        delta = max(delta, abs(x - new_x));
      }

      undefinedGamma -= (totalCount - uniqueTermsCount) / denominator;
      denominator = newDenominator;
    }
  }

  /**
   * Finds optimal alpha for dirichlet process
   *
   * \alpha = \argmax_{\alpha} \sum_{t=1}^T I{t is new} * \log{\frac{\alpha}{\alpha + t - 1}} + I{t is old} * \log{\frac{t - 1}{\alpha + t - 1}}
   *
   * @return new optimal alpha for dirichlet process
   *
   * TODO: calculate minDpAlpha and maxDpAlpha
   */
  private double findDpAlpha() {
    final double minDpAlpha = 1e-3;
    final double maxDpAlpha = totalCount;

    return MathTools.bisection(minDpAlpha, maxDpAlpha, new AnalyticFunc.Stub() {
      @Override
      public double value(double alpha) {
        double result = 0;

        for (int i = 0, it = 0; i < totalCount; ++i) {
          if (it < generationIndices.size() && generationIndices.get(it) < i) {
            ++it;
          }

          if (it < generationIndices.size() && i == generationIndices.get(it)) {
            result += i / (alpha * (alpha + i));
          } else {
            result += -1 / (alpha + i);
          }
        }

        return result;
      }
    });
  }

  public void applyUpdates(final double alpha) {
    // if nothing to apply
    if (newTermsTotalCount == 0) {
      return;
    }

    // update count values
    totalCount += newTermsTotalCount;

    TIntIntIterator it = newTermsCount.iterator();
    while (it.hasNext()) {
      count.set(it.key(), count.get(it.key()) + it.value());
      it.advance();
    }

    dpAlpha = findDpAlpha();
    gradientDescentGamma(alpha);

    newTermsTotalCount = 0;
    newTermsCount.clear();
  }


  /*
  public void update(TIntIntMap weightsPools, double alpha) {
    alpha /= poolSize;

    final int[] updates = weightsPools.keys();
    final int[] count = weightsPools.values();
    { // gradientDescentGamma step
      double gradientTermSum = 0;
      double[] oldValues = new double[updates.length];
      for (int i = 0; i < updates.length; i++) {
        final int windex = updates[i];
        double value = oldValues[i] = count.get(windex);
        double pAB = exp(value + undefined) / denominator;
        if (abs(pAB) < 1e-10)
          continue;
        gradientTermSum += count[i];
      }
      if (gradientTermSum == 0)
        return;
      final double newUndefined = undefined - alpha * gradientTermSum * pGen(-1);

      final VecIterator it = count.nonZeroes();
      int nzCount = 0;
      double newDenominator = 1;
      while (it.advance()) { // updating all non zeroes as if they were negative, then we will change the gradientDescentGamma for positives
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
        count.set(windex, value - newUndefined);
      }
      newDenominator += exp(undefined) * (count.dim() - nzCount);

      undefined = newUndefined;
      if (newDenominator < 0)
        System.out.println();
      denominator = newDenominator;
    }
  }
  */

  // TODO: implement
  public double pGen(int windex) {
    double logPGenB = 0;

    if (!Double.isNaN(dpAlpha)) { // DP model
      final double statPower = denominator - count.dim();

      if (windex < 0 || windex >= count.dim() || count.get(windex) < MathTools.EPSILON)
        // log(\frac{1}{N - m + 1} {\alpha \over \alpha + n - 1}),
        // where N -- vocabulary size, m -- number of words, occurred in statistics, \alpha -- DP expansion parameter, n -- statistics power
        logPGenB = -log(count.dim() - count.size() + 1) + log(dpAlpha) - log(dpAlpha + statPower - 1);
      else
        // log({\e^s_i \over \alpha + n - 1}),
        // where N -- vocabulary size, m -- number of words, occurred in statistics, \alpha -- DP expansion parameter, n -- statistics power
        logPGenB = log(count.get(windex) - 1) - log(dpAlpha + statPower - 1);
    }

    return exp(logPGenB);
  }

  private static final ThreadLocal<ObjectMapper> mapper = ThreadLocal.withInitial(ObjectMapper::new);

  // TODO: refactor, review
  public void visitVariants(final TIntDoubleProcedure todo) {
    final VecIterator it = count.nonZeroes();
    while (it.advance()) {
      todo.execute(it.index(), pGen(it.index()));
    }
  }

  // TODO: refactor, review
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

  // TODO: refactor, review
  public void print(final Dictionary<CharSeq> words,
                    final Writer to,
                    final boolean limit) {
    if (poissonLambdaCount < MINIMUM_STATISTICS_TO_OUTPUT && limit) {
      return;
    }

    final ObjectNode output = mapper.get().createObjectNode();
    output.put("poissonSum", poissonLambdaSum);
    output.put("poissonCount", poissonLambdaCount);
    output.put("undefined", undefinedGamma);
    output.put("denominator", denominator);

    final ObjectNode wordsNode = output.putObject("words");
    final int[] myIndices = new int[count.size() + 1];
    final double[] myProbabs = new double[count.size() + 1];

    int index = 0;

    final VecIterator nz = count.nonZeroes();
    while (nz.advance()) {
      myIndices[index] = nz.index();
      myProbabs[index] = pGen(nz.index());
      index++;
    }

    ArrayTools.parallelSort(myProbabs, myIndices);
    int wordsCount = 0;
    for (int i = myIndices.length - 1; i >= 0; i--) {
      if (myProbabs[i] < 1e-6 && limit) {
        break;
      }

      final int windex = myIndices[i];
      wordsCount++;

      if (words.size() > windex) {
        wordsNode.put(words.get(windex).toString(), myProbabs[i]);
      } else {
        wordsNode.put(SimpleGenerativeModel.EMPTY_ID, myProbabs[i]);
      }
    }

    if (wordsCount < 2) {
      return;
    }

    final ObjectWriter writer = mapper.get().writerWithDefaultPrettyPrinter();
    try {
      if (providerIndex < dict.size()) {
        to.append(dict.get(providerIndex).toString()).append(": ");
      } else {
        to.append(SimpleGenerativeModel.EMPTY_ID).append(": ");
      }

      to.append(writer.writeValueAsString(output));
      to.append("\n");
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }


  // TODO: refactor, review
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
      providerIndex = dict.search(phraseBuilder.build());
    }
    final ObjectReader reader = mapper.get().reader();
    count = new SparseVec(dict.size());

    try {
      final JsonNode node = reader.readTree(json);
      poissonLambdaSum = node.get("poissonSum").asDouble();
      poissonLambdaCount = node.get("poissonCount").asDouble();
      denominator = node.get("denominator").asDouble();
      undefinedGamma = node.get("undefined").asDouble();
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
        count.set(dict.search(phraseBuilder.build()), weight);
        phraseBuilder.clear();

      }

      { // DP schema
        dpAlpha = optimalExpansionDP(totalWeight, count.size());
      }
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  private String termToText(final int index,
                            final Dictionary<CharSeq> dict) {
    if (index < 0 || index >= dict.size()) {
      return SimpleGenerativeModel.EMPTY_ID;
    }

    return dict.get(index).toString();
  }
}
