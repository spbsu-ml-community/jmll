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
  public static boolean DEBUG = false;
  public static final int MINIMUM_STATISTICS_TO_OUTPUT = 50;

  private static final int POOL_SIZE = 30;

  private static Pattern headerPattern = Pattern.compile("(\\[(?:[^, ]+(?:, )?)*\\]): \\{(.*)");

  private Dictionary<CharSeq> dict;

  private double poissonLambdaSum = 1;
  private double poissonLambdaCount = 1;

  private int updatesCount = 1; // TODO: debug

  private double dpAlpha = 1.0; // TODO: think about good initialization
  private double undefinedGamma = 0;
  private double denominator;

  private SparseVec count;
  private SparseVec gamma;

  private int totalCount; // TODO: save to get real dpAlpha
  private int uniqueTermsCount; // TODO: save to get real dpAlpha
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
  private double logProbabilityOfAllTerms(final double alpha) {
    double result = 0;

    for (int i = 0, it = 0; i < allTerms.size(); ++i) {
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
   * @param variant  the mask of generated terms by the current term
   * @param fragment the sequence of terms
   * @return log-probability of fragment generation by the current term
   */
  public double logP(final int variant,
                     final IntSeq fragment) {
    double result = MathTools.logPoissonProbability(poissonLambdaSum / poissonLambdaCount, Integer.bitCount(variant));

    if (variant == 0) {
      return result;
    }

    int n = totalCount;
    int uniqueCount = uniqueTermsCount;
    double denominator = this.denominator;

    for (int i = 0, mask = variant; i < fragment.length(); i++, mask >>= 1) {
      if ((mask & 1) == 0) {
        continue; // skip not generated term
      }

      ++n; // met new generated word
      final int index = fragment.intAt(i);

      if (count.get(index) == 0) {
        result += -log(dict.size() - uniqueCount) + log(dpAlpha) - log(dpAlpha + n - 1);
      } else {
        result += gamma.get(index) - log(denominator) + log(n - 1) - log(dpAlpha + n - 1);
      }
    }

    return result;
  }

  /**
   * Returns the log-probability of term generation by the current term
   *
   * @param index index of the generated term
   * @return log-probability of term generation by the current term
   */
  public double logP(final int index) {
    final int n = totalCount + 1;
    double result = MathTools.logPoissonProbability(poissonLambdaSum / poissonLambdaCount, 1);

    if (count.get(index) == 0) {
      return result - log(dict.size() - uniqueTermsCount) + log(dpAlpha) - log(dpAlpha + n - 1);
    } else {
      return result + gamma.get(index) - log(denominator) + log(n - 1) - log(dpAlpha + n - 1);
    }
  }

  /**
   * Updates model:
   * - affects poisson distribution
   * - applies updates if variant includes new term
   * - applies updates if provider has a lot of updates
   *
   * @param variant the mask of generated terms by the current term
   * @param seq     the sequence of terms
   * @param alpha   descent parameter
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

        generationIndices.add(totalCount);

        gamma.set(index, undefinedGamma);
        denominator += exp(undefinedGamma);

        ++uniqueTermsCount;
        ++totalCount;
        count.adjust(index, 1);

        dpAlpha = findDpAlphaFast();
      } else {
        newTerms.add(index);
      }
    }

    // TODO: review
    if (!DEBUG && newTerms.size() < updatesCount * POOL_SIZE) {
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
    Utils.Timer.start("gradient", false);

    for (int term : newTerms) {
      gradientDescentStep(alpha, term);
    }

    Utils.Timer.stop("gradient", false);
  }

  /**
   * Simulates one gradient descent step
   *
   * @param alpha descent parameter
   * @param term  index of the new generated term
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

    undefinedGamma -= alpha * exp(undefinedGamma) / denominator;
    denominator = newDenominator;
  }

  /**
   * Finds optimal alpha for dirichlet process
   * \alpha = \argmax_{\alpha} \sum_{t=1}^T I{t is new} * \log{\frac{\alpha}{\alpha + t - 1}} + I{t is old} * \log{\frac{t - 1}{\alpha + t - 1}}
   *
   * @return new optimal alpha for dirichlet process
   * <p>
   * TODO: calculate minDpAlpha and maxDpAlpha
   * TODO: may be stochastic
   */
  private double findDpAlphaSlow() {
    if (totalCount == 1) {
      return dpAlpha; // not enough data
    }

    Utils.Timer.start("alpha", false);

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

    Utils.Timer.stop("alpha", false);

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
   * Finds alpha from the equation below:
   * m = \alpha log(1 + \frac{n}{\alpha})
   * where n -- terms total count, m -- unique terms found
   *
   * @return new optimal alpha for dirichlet process
   */
  private double findDpAlphaFast() {
    Utils.Timer.start("alpha", false);

    final double alpha = MathTools.bisection(0, uniqueTermsCount, new AnalyticFunc.Stub() {
      @Override
      public double value(double x) {
        return x * log(1 + totalCount / x) - uniqueTermsCount;
      }
    });

    Utils.Timer.stop("alpha", false);
    return alpha;
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

    ++updatesCount;

    // update count values
    for (int term : newTerms) {
      count.adjust(term, 1);
      ++totalCount;
    }

    dpAlpha = findDpAlphaFast();
    gradientDescent(alpha);

    newTerms.clear();
  }

  private static final ThreadLocal<ObjectMapper> mapper = ThreadLocal.withInitial(ObjectMapper::new);

  public void print(final Writer to,
                    final boolean limit) {
    if (poissonLambdaCount < MINIMUM_STATISTICS_TO_OUTPUT && limit) {
      return;
    }

    final ObjectNode output = mapper.get().createObjectNode();

    output.put("poissonSum", poissonLambdaSum);
    output.put("poissonCount", poissonLambdaCount);
    output.put("undefined", undefinedGamma);

    final ObjectNode wordsNode = output.putObject("words");

    int index = 0;
    final int[] indices = new int[count.size() + 1];
    final double[] probabilities = new double[count.size() + 1];

    final VecIterator nz = count.nonZeroes();
    while (nz.advance()) {
      indices[index] = nz.index();
      probabilities[index] = exp(gamma.get(nz.index())) / denominator;
      ++index;
    }

    ArrayTools.parallelSort(probabilities, indices);

    for (int i = indices.length - 1; i >= 0; i--) {
      if (probabilities[i] < 1e-6 && limit) {
        break;
      }

      final ObjectNode node = wordsNode.putObject(termToText(indices[i], dict));
      node.put("prob", probabilities[i]);
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

  public WordGenProbabilityProvider(final CharSequence presentation,
                                    final Dictionary<CharSeq> dictionary) {
    final String json;
    {
      final SeqBuilder<CharSeq> phraseBuilder = new ArraySeqBuilder<>(CharSeq.class);
      final Matcher matcher = headerPattern.matcher(presentation);
      if (!matcher.find()) {
        throw new IllegalArgumentException(presentation.toString());
      }

      final String phrase = matcher.group(1);
      final CharSequence[] parts = CharSeqTools.split(phrase.subSequence(1, phrase.length() - 1), ", ");
      for (final CharSequence part : parts) {
        phraseBuilder.add(CharSeq.create(part.toString()));
      }
      json = "{" + matcher.group(2) + "}";

      int index = dictionary.size();
      try {
        index = dictionary.search(phraseBuilder.build());
      } catch (RuntimeException e) {
        // empty word
      }

      providerIndex = index;
    }
    final ObjectReader reader = mapper.get().reader();

    count = new SparseVec(dictionary.size(), 0); // without empty word
    gamma = new SparseVec(dictionary.size(), 0); // parameters to learn
    denominator = 1;

    try {
      final JsonNode node = reader.readTree(json);

      poissonLambdaSum = node.get("poissonSum").asDouble();
      poissonLambdaCount = node.get("poissonCount").asDouble();
      undefinedGamma = node.get("undefined").asDouble();

      final JsonNode words = node.get("words");
      final Iterator<Map.Entry<String, JsonNode>> wordsIt = words.fields();
      final SeqBuilder<CharSeq> phraseBuilder = new ArraySeqBuilder<>(CharSeq.class);

      while (wordsIt.hasNext()) {
        final Map.Entry<String, JsonNode> next = wordsIt.next();
        final String phrase = next.getKey();
        final JsonNode value = next.getValue();

        final CharSequence[] parts = CharSeqTools.split(phrase.subSequence(1, phrase.length() - 1), ", ");
        for (final CharSequence part : parts) {
          phraseBuilder.add(CharSeq.create(part.toString()));
        }

        final int index = dictionary.search(phraseBuilder.build());
        final int countValue = value.get("count").asInt();
        final double gammaValue = value.get("gamma").asDouble();

        count.set(index, countValue);
        gamma.set(index, gammaValue);

        denominator += exp(gammaValue);
        totalCount += countValue;
        ++uniqueTermsCount;

        phraseBuilder.clear();
      }

      final JsonNode indices = node.get("indices");
      for (final JsonNode indexNode : indices) {
        generationIndices.add(indexNode.asInt());
      }

      if (totalCount == 0) {
        return; // no statistics to process
      }

      dpAlpha = findDpAlphaFast();
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  public void visitVariants(final TIntDoubleProcedure todo) {
    final VecIterator it = count.nonZeroes();
    while (it.advance()) {
      todo.execute(it.index(), logP(it.index()));
    }

  }

  private String termToText(final int index,
                            final Dictionary<CharSeq> dict) {
    if (index < 0 || index >= dict.size()) {
      return "[" + SimpleGenerativeModel.EMPTY_ID + "]";
    }

    return dict.get(index).toString();
  }
}
