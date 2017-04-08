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
  private double poissonLambdaSum = 1;
  private double poissonLambdaCount = 1;
  private double denominator;

  // TODO: remove
  private double undefined;

  private static Pattern headerPattern = Pattern.compile("(\\[(?:[^, ]+(?:, )?)*\\]): \\{(.*)");

  private double dpAlpha = Double.NaN;

  SparseVec beta;
  SparseVec gamma;

  private int totalTermsCount;
  private int uniqueTermsCount;

  final int providerIndex; // for debug purposes

  // TODO: is it necessary?
  double probab;

  // TODO: is it necessary?
  private Dictionary<CharSeq> dict;

  // TODO: is it necessary?
  private WordGenProbabilityProvider[] all;

  private int poolSize;
  private final List<Integer> termsOccurrences = new ArrayList<>();
  private final TIntIntMap newTermsCount = new TIntIntHashMap();

  public WordGenProbabilityProvider(final int dim,
                                    final int wordIndex) {
    providerIndex = wordIndex;
    beta = new SparseVec(dim, 0); // without empty word

    gamma = new SparseVec(dim, 0); // parameters to learn
    denominator = 1 + dim; // 1 + \sum_i e^gamma[i] = 1 + \sum_i e^0
  }


  /** Returns the log-probability of fragment generation by the current term
   * p = Poisson(\lambda) * p(fragment | current term)
   *
   * If the given fragment is empty, p(fragment | current term) = \frac{1}{1 + \sum_i e^gamma[i]} * \frac{n - 1}{\alpha + n - 1}
   * Otherwise, p(fragment | current term) = \prod_{t \in fragment} p(t | current term)
   *
   * If t is new class, p(t | current term) = \frac{1}{N - m} * \frac{\alpha}{\alpha + n - 1}
   * Otherwise, p(t | current term) = \frac{e^gamma[t]}{1 + \sum_i e^gamma[i]} * \frac{n - 1}{\alpha + n - 1}
   *
   * Where:
   * n -- the number of the current word
   * N -- total count of classes (size of the dictionary)
   * m -- count of classes for the current term
   *
   * Additional description:
   * denominator = 1 + \sum_i e^gamma[i]
   *
   * @param variant the mask of generated terms
   * @param fragment the sequence of terms
   * @return log-probability of fragment generation by the current term
   */
  public double logP(int variant,
                     final IntSeq fragment) {
    int n = totalTermsCount;
    int uniqueCount = uniqueTermsCount;

    double result = MathTools.logPoissonProbability(poissonLambdaSum / poissonLambdaCount, Integer.bitCount(variant));

    if (variant == 0) {
      result += -log(denominator) + log(n) - log(dpAlpha + n);
    } else {
      final List<Integer> newTerms = new LinkedList<>();

      for (int i = 0; i < fragment.length(); i++, variant >>= 1) {
        if ((variant & 1) == 0) {
          continue; // skip not generated term
        }

        ++n; // met new generated word
        final int termIndex = fragment.intAt(i);

        boolean isNewTerm = false;
        if (beta.get(termIndex) == 0) {
          isNewTerm = newTerms.indexOf(termIndex) == -1;
        }

        if (isNewTerm) {
          result += -log(dict.size() - uniqueCount) + log(dpAlpha) - log(dpAlpha + n - 1);
          ++uniqueCount;
          newTerms.add(termIndex);
        } else {
          result += gamma.get(termIndex) - log(denominator) + log(n - 1) - log(dpAlpha + n - 1);
        }
      }
    }

    return result;
  }


  // TODO: merge with constructor
  public void init(final WordGenProbabilityProvider[] providers,
                   final Dictionary<CharSeq> dictionary) {
    all = providers;
    dict = dictionary;

    uniqueTermsCount = 0;

    final VecIterator nz = beta.nonZeroes();
    while (nz.advance()) {
      final int count = (int) nz.value();
      totalTermsCount += count;
      uniqueTermsCount++;
    }
  }


  public void update(int variant,
                     final IntSeq seq,
                     final double alpha) {
    final int length = seq.length();

    poissonLambdaCount++;
    poissonLambdaSum += Integer.bitCount(variant);

    if (DEBUG) {
      System.out.print(wordText(providerIndex, dict) + "->");
    }

    for (int i = 0; i < length; ++i, variant >>= 1) {
      if ((variant & 1) == 0) {
        continue; // skip not generated term
      }

      final int termIndex = seq.intAt(i);
      termsOccurrences.add(termIndex);
      newTermsCount.adjustOrPutValue(termIndex, 1, 1);

      if (DEBUG) {
        System.out.print(" " + wordText(termIndex, dict));
      }
    }

    poolSize++;
    if (!DEBUG && poolSize < poissonLambdaCount / 20.) {
      return;
    }

    applyUpdates(alpha);
  }

  private double gradientDescentGamma(final List<Integer> termsOccurrences) {
    // TODO: iplement
    return 0.0;
  }

  /**
   * Finds optimal alpha for dirichlet process
   *
   * @param termsCount the number of terms
   * @param newTermsIndices the indices of new terms
   * @return new optimal alpha for dirichlet process
   *
   * TODO: calculate minDpAlpha and maxDpAlpha
   * TODO: replace ArrayList<Integer>
   * TODO: totalTermsCount + i replace with counter
   */
  private double findNewDpAlpha(final int termsCount,
                                final ArrayList<Integer> newTermsIndices) {
    final double minDpAlpha = 1e-3;
    final double maxDpAlpha = totalTermsCount;

    return MathTools.bisection(minDpAlpha, maxDpAlpha, new AnalyticFunc.Stub() {
      @Override
      public double value(double x) {
        double result = 0;

        for (int i = 0, it = 0; i < termsCount; ++i) {
          if (it < newTermsIndices.size() && newTermsIndices.get(it) < i) {
            ++it;
          }

          // the number of the current word = totalTermsCount + i + 1
          if (it < newTermsIndices.size() && i == newTermsIndices.get(it)) {
            result += (totalTermsCount + i) / (dpAlpha * (dpAlpha + totalTermsCount + i));
          } else {
            result += -1 / (dpAlpha + totalTermsCount + i);
          }
        }

        return result;
      }
    });
  }

  public void applyUpdates(final double alpha) {
    final ArrayList<Integer> newTermsIndices = new ArrayList<>();

    for (int index : termsOccurrences) {
      if (beta.get(index) == 0) {
        newTermsIndices.add(index);
      }

      beta.set(index, beta.get(index) + 1);
    }

    dpAlpha = findNewDpAlpha(termsOccurrences.size(), newTermsIndices);
    // gradientDescentGamma(alpha);


    termsOccurrences.clear();
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
        beta.set(windex, value - newUndefined);
      }
      newDenominator += exp(undefined) * (beta.dim() - nzCount);

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

    return exp(logPGenB);
  }

  private static final ThreadLocal<ObjectMapper> mapper = ThreadLocal.withInitial(ObjectMapper::new);

  // TODO: refactor, review
  public void visitVariants(final TIntDoubleProcedure todo) {
    final VecIterator it = beta.nonZeroes();
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
    output.put("undefined", undefined);
    output.put("denominator", denominator);

    final ObjectNode wordsNode = output.putObject("words");
    final int[] myIndices = new int[beta.size() + 1];
    final double[] myProbabs = new double[beta.size() + 1];

    int index = 0;

    final VecIterator nz = beta.nonZeroes();
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

  // TODO: refactor, review
  private String wordText(final int index,
                          final Dictionary<CharSeq> dict) {
    if (index < 0 || index >= dict.size()) {
      return SimpleGenerativeModel.EMPTY_ID;
    }

    // TODO: is it bug?
    this.dict = dict;

    return dict.get(index).toString();
  }
}
