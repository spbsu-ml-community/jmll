package com.spbsu.direct.gen;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.ObjectWriter;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.spbsu.commons.io.codec.seq.Dictionary;
import com.spbsu.commons.math.AnalyticFunc;
import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.math.vectors.VecIterator;
import com.spbsu.commons.math.vectors.impl.vectors.SparseVec;
import com.spbsu.commons.seq.*;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.direct.Utils;
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

  private static final int POOL_SIZE = 20;
  private static final int BUFFER_SIZE = 10;

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
  private final ArrayList<Integer> allTerms = new ArrayList<>();
  private final ArrayList<Integer> newTerms = new ArrayList<>();
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
   * Returns the log-probability of generation of all terms by the current term
   *
   * @return the log-probability of generation of all terms
   */
  private double logProbabilityOfAllTerms() {
    double result = 0;

    for (int i = 0, it = 0; i < allTerms.size(); ++i) {
      if (it < generationIndices.size() && generationIndices.get(it) < i) {
        ++it;
      }

      if (it < generationIndices.size() && i == generationIndices.get(it)) {
        result += -log(dict.size() - it) + log(dpAlpha) - log(dpAlpha + i);
      } else {
        result += gamma.get(allTerms.get(i)) - log(denominator) + log(i) - log(dpAlpha + i);
      }
    }

    return result;
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
   * @param variant the mask of generated terms by the current term
   * @param fragment the sequence of terms
   * @return log-probability of fragment generation by the current term
   */
  public double logP(final int variant,
                     final IntSeq fragment) {
    int n = totalCount;
    int uniqueCount = uniqueTermsCount;
    double denominator = this.denominator;

    double result = MathTools.logPoissonProbability(poissonLambdaSum / poissonLambdaCount, Integer.bitCount(variant));

    int[] newTerms = new int[BUFFER_SIZE];
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

  /**
   * Updates model:
   * - affects poisson distribution
   * - applies updates if variant includes new term
   * - applies updates if provider has a lot of updates
   *
   * @param variant the mask of generated terms by the current term
   * @param seq the sequence of terms
   * @param alpha descent parameter
   */
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
      allTerms.add(index);

      if (count.get(index) == 0) {
        applyUpdates(alpha); // new term -- need to apply previous changes

        gamma.set(index, undefinedGamma);
        denominator += exp(undefinedGamma);

        ++uniqueTermsCount;
        count.adjust(index, 1);

        generationIndices.add(totalCount++);

        dpAlpha = findDpAlpha();
      } else {
        newTerms.add(index);
      }
    }

    // TODO: review
    if (!DEBUG && newTerms.size() / poissonLambdaCount < POOL_SIZE) {
      return;
    }

    applyUpdates(alpha);
  }

  /**
   * Simulates gradient descent steps one by one for all new terms
   *
   * @param alpha descent parameter
   */
  private void gradientDescent(double alpha) {
    alpha /= totalCount; // TODO: is it okay?

    for (int term : newTerms) {
      gradientDescentStep(alpha, term);
    }
  }

  /**
   * Simulates one gradient descent step
   *
   * @param alpha descent parameter
   * @param term index of the new generated term
   */
  private void gradientDescentStep(final double alpha,
                                   final int term) {
    double newDenominator = 1;

    VecIterator it = count.nonZeroes();
    while (it.advance()) {
      int index = it.index();

      double x = gamma.get(index);
      double gradient = (index == term ? 1 : 0) - exp(x) / denominator;
      double new_x = x + alpha * gradient;

      gamma.set(index, new_x);
      newDenominator += exp(new_x);
    }

    undefinedGamma -= 1 / denominator;
    denominator = newDenominator;
  }

  /**
   * Finds optimal alpha for dirichlet process
   * <p>
   * \alpha = \argmax_{\alpha} \sum_{t=1}^T I{t is new} * \log{\frac{\alpha}{\alpha + t - 1}} + I{t is old} * \log{\frac{t - 1}{\alpha + t - 1}}
   *
   * @return new optimal alpha for dirichlet process
   * <p>
   * TODO: calculate minDpAlpha and maxDpAlpha
   * TODO: may be stochastic
   */
  private double findDpAlpha() {
    if (totalCount == 1) {
      return dpAlpha; // not enough data
    }

    final double minDpAlpha = 1e-3;
    final double maxDpAlpha = 10_000;

    double alpha = dpAlpha;

    while (alpha >= minDpAlpha && alpha <= maxDpAlpha) {
      final double prob = getAlpha(alpha);
      final double grad = getAlphaGradient(alpha);
      alpha += grad / totalCount;
      final double newProb = getAlpha(alpha);

      if (abs(newProb - prob) < 1e-4) {
        break;
      }
    }

    alpha = Math.max(minDpAlpha, Math.min(alpha, maxDpAlpha));
    return alpha;
  }

  private double getAlpha(double alpha) {
    double result = 0;

    for (int i = 0, it = 0; i < totalCount; ++i) {
      if (it < generationIndices.size() && generationIndices.get(it) < i) {
        ++it;
      }

      if (it < generationIndices.size() && i == generationIndices.get(it)) {
        result += -log(dict.size() - it) + log(alpha) - log(alpha + i);
      } else {
        result += gamma.get(allTerms.get(i)) - log(denominator) + log(i) - log(alpha + i);
      }
    }

    return result;
  }

  private double getAlphaGradient(double alpha) {
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


  /**
   * Applies updates:
   * - updates words count
   * - finds new dpAlpha
   * - runs gradient descent
   *
   * @param alpha descent parameter
   */
  public void applyUpdates(final double alpha) {
    // if nothing to apply
    if (newTerms.isEmpty()) {
      return;
    }

    // update count values
    for (int term : newTerms) {
      count.adjust(term, 1);
      ++totalCount;
    }

    // Utils.Timer.start(String.format("Before  (dpAlpha = %f, prob = %f)", dpAlpha, logProbabilityOfAllTerms()));
    dpAlpha = findDpAlpha();
    gradientDescent(alpha);
    // Utils.Timer.stop(String.format("After (dpAlpha = %f, prob = %f)", dpAlpha, logProbabilityOfAllTerms()));

    newTerms.clear();
  }

  private static final ThreadLocal<ObjectMapper> mapper = ThreadLocal.withInitial(ObjectMapper::new);

  // TODO: refactor, review
  public void visitVariants(final TIntDoubleProcedure todo) {
    final VecIterator it = count.nonZeroes();
    /*while (it.advance()) {
      todo.execute(it.index(), pGen(it.index()));
    }*/
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

  public void print(final Writer to,
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

    int index = 0;
    final int[] indices = new int[count.size() + 1];
    final double[] probabilities = new double[count.size() + 1];

    final VecIterator nz = count.nonZeroes();
    while (nz.advance()) {
      indices[index] = nz.index();
      probabilities[index] = gamma.get(nz.index());
      ++index;
    }

    ArrayTools.parallelSort(probabilities, indices);

    for (int i = indices.length - 1; i >= 0; i--) {
      if (probabilities[i] < 1e-6 && limit) {
        break;
      }

      final ObjectNode node = wordsNode.putObject(termToText(indices[i], dict));
      node.put("prob", gamma.get(indices[i]));
      node.put("gamma", gamma.get(indices[i]));
      node.put("count", count.get(indices[i]));
    }

    final ArrayNode indicesNode = output.putArray("indices");
    for (int i : generationIndices) {
      indicesNode.add(i);
    }

    final ObjectWriter writer = mapper.get().writerWithDefaultPrettyPrinter();
    try {
      to.append(termToText(providerIndex, dict))
              .append(": ")
              .append(writer.writeValueAsString(output))
              .append("\n");
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }


  // TODO: refactor, review, implement
  public WordGenProbabilityProvider(final CharSequence presentation,
                                    final Dictionary<CharSeq> dict) {
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
      undefinedGamma = node.get("undefined").asDouble();
      denominator = node.get("denominator").asDouble();

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
