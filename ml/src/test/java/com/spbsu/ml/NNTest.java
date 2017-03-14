package com.spbsu.ml;

import com.spbsu.commons.func.Action;
import com.spbsu.commons.func.Processor;
import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.math.Trans;
import com.spbsu.commons.math.TransC1;
import com.spbsu.commons.math.io.Vec2CharSequenceConverter;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.MxTools;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.seq.*;
import com.spbsu.ml.data.tools.Pool;
import com.spbsu.ml.data.tools.PoolByRowsBuilder;
import com.spbsu.ml.func.generic.Log;
import com.spbsu.ml.func.generic.ParallelFunc;
import com.spbsu.ml.func.generic.WSum;
import com.spbsu.ml.loss.CompositeFunc;
import com.spbsu.ml.loss.DSSumFuncComposite;
import com.spbsu.ml.loss.LL;
import com.spbsu.ml.loss.blockwise.BlockwiseMLL;
import com.spbsu.ml.meta.DataSetMeta;
import com.spbsu.ml.meta.FeatureMeta;
import com.spbsu.ml.meta.items.FakeItem;
import com.spbsu.ml.meta.items.QURLItem;
import com.spbsu.ml.methods.StochasticGradientDescent;
import com.spbsu.ml.models.nn.LayeredNetwork;
import com.spbsu.ml.models.nn.NeuralSpider;
import com.spbsu.ml.models.nn.nfa.NFANetwork;
import com.spbsu.ml.models.nn.nfa.NFANetworkWithLZW;
import com.spbsu.ml.models.nn.nfa.SeqWeightsCalculator;
import com.spbsu.ml.models.nn.nfa.WeightsCalculator;
import com.spbsu.ml.testUtils.TestResourceLoader;
import org.jetbrains.annotations.NotNull;
import org.junit.Assert;
import org.junit.Test;

import java.io.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.zip.GZIPInputStream;

/**
 * User: solar
 * Date: 25.05.15
 * Time: 16:39
 */
public abstract class NNTest {
  private final FastRandom rng = new FastRandom(0);
  private Pool<QURLItem> featuresTxtPool;

  public NNTest() throws IOException {
    featuresTxtPool = (Pool<QURLItem>) TestResourceLoader.loadPool("features.txt.gz");
  }

  @Test
  public void testValue() {
    final LayeredNetwork nn = new LayeredNetwork(rng, 0, 3, 3, 3, 1);
    final Vec weights = VecTools.fill(new ArrayVec(nn.dim()), 1);
    final Vec vec = nn.compute(new ArrayVec(0, 1, 1), weights);
    Assert.assertEquals(1, vec.dim());
    Assert.assertEquals(0.9427, vec.get(0), 0.0001);
  }

  @Test
  public void testGradient() {
    final LayeredNetwork nn = new LayeredNetwork(rng, 0., 3, 3, 3, 1);
    final Vec weights = VecTools.fill(new ArrayVec(nn.dim()), 1);
    final Vec vec = nn.parametersGradient(new ArrayVec(0, 1, 1), new Log(1, 0), weights);
    Assert.assertEquals(nn.dim(), vec.dim());
    Assert.assertTrue(VecTools.distance(vec, new Vec2CharSequenceConverter().convertFrom("24 0 0 0 0 0.00112 0.00112 0 0.00112 0.00112 0 0.00112 0.00112 0.00313 0.00313 0.00313 0.00313 0.00313 0.00313 0.00313 0.00313 0.00313 0.05348 0.05348 0.05348\n")) < MathTools.EPSILON * vec.dim());
  }

  @Test
  public void testConvergence() {
    final PoolByRowsBuilder<FakeItem> pbuilder = new PoolByRowsBuilder<>(DataSetMeta.ItemType.FAKE);
    pbuilder.allocateFakeFeatures(3, FeatureMeta.ValueType.VEC);
    pbuilder.allocateFakeTarget(FeatureMeta.ValueType.INTS);
    for (int i = 0; i < 10000; i++) {
      final Vec next = new ArrayVec(3);
      for (int j = 0; j < next.dim(); j++)
        next.set(j, rng.nextInt(2));
      pbuilder.setFeatures(0, next);
      pbuilder.setTarget(0, (int) next.get(0));
      pbuilder.nextItem();
    }

    final Pool<FakeItem> pool = pbuilder.create();
    final LayeredNetwork network = new LayeredNetwork(rng, 0., 3, 3, 3, 1);
    final StochasticGradientDescent<FakeItem> gradientDescent = new StochasticGradientDescent<FakeItem>(rng, 4, 1000, 0.8) {
      public void init(Vec cursor) {
        VecTools.fillUniform(cursor, rng);
      }
    };
    final Mx data = pool.vecData().data();
    final LL ll = pool.target(LL.class);
    final DSSumFuncComposite<FakeItem> target = new DSSumFuncComposite<>(pool.data(), ll, argument -> {
      final Vec row = data.row(argument.id);
      return network.decisionByInput(row);
    });
    final DSSumFuncComposite<FakeItem>.Decision decision = gradientDescent.fit(pool.data(), target);
    System.out.println(decision.x);
    final Vec vals = new ArrayVec(pool.size());
    for (int i = 0; i < vals.length(); i++) {
      vals.set(i, decision.compute(pool.data().at(i)).get(0));
    }
    System.out.println(Math.exp(-ll.value(vals) / ll.dim()));
    Assert.assertTrue(1.1 > Math.exp(-ll.value(vals) / ll.dim()));
  }

  @Test
  public void testValueSeq() {
    final NFANetwork<Character> nfa = new NFANetwork<>(rng, 0., 5, 1, CharSeq.create("ab"));
    final Trans aba = nfa.decisionByInput(CharSeq.create("aba"));
    Assert.assertEquals(0.2 + 0.16 + 0.128, aba.trans(new ArrayVec(nfa.dim())).get(1), 0.0001);
  }

  @Test
  public void testSeqGradient1() {
    String message = "\n";
    final NFANetwork<Character> nfa = new NFANetwork<>(rng, 0., 2, 1, CharSeq.create("ab"));

    final NeuralSpider.NeuralNet ab = nfa.decisionByInput(CharSeq.create("ab"));
    final NeuralSpider.NeuralNet ba = nfa.decisionByInput(CharSeq.create("ba"));
    final Vec x = new ArrayVec(1., 2.);
    message += nfa.ppState(ab.state(x), CharSeq.create("ab"));
    { // Positive
      final CompositeFunc target = new CompositeFunc(new WSum(new ArrayVec(0, 1)), new ParallelFunc(2, new Log(1., 0.)));
      final Vec gradientAb = ab.gradientTo(x, new ArrayVec(2), target);
      message += nfa.ppState(ba.state(x), CharSeq.create("ba"));
      final Vec gradientBa = ba.gradientTo(x, new ArrayVec(2), target);
      // composite result:  1/(1+e^x)*(1 + e^x/(1+e^y))
      message += "or: " + x + "\n"
          + "ab: " + gradientAb + "\n"
          + "ba: " + gradientBa + "\n";
      // composite gradient by x: -e^x/(1+e^x)*1/(1+e^x)*e^y/(1+e^y)
      Assert.assertEquals(message, -0.26894, gradientAb.get(0), 0.00001);
      // composite gradient by x: -e^y/(1+e^y)*1/(1+e^y)*1/(1+e^x)
      Assert.assertEquals(message, -0.1192, gradientAb.get(1), 0.00001);
      Assert.assertTrue(message, VecTools.equals(gradientAb, gradientBa));
    }
    { // Negative
      final CompositeFunc target = new CompositeFunc(new WSum(new ArrayVec(1, 0)), new ParallelFunc(2, new Log(1., 0.)));
      final Vec gradientAb = ab.gradientTo(x, new ArrayVec(2), target);
      message += nfa.ppState(ba.state(x), CharSeq.create("ba"));
      final Vec gradientBa = ba.gradientTo(x, new ArrayVec(2), target);
      // composite result:  1/(1+e^x)*(1 + e^x/(1+e^y))
      message += "or: " + x + "\n"
          + "ab: " + gradientAb + "\n"
          + "ba: " + gradientBa + "\n";
      // composite gradient by x: -e^x/(1+e^x)*1/(1+e^x)*e^y/(1+e^y)
      Assert.assertEquals(message, 0.26894, gradientAb.get(0), 0.00001);
      // composite gradient by x: -e^y/(1+e^y)*1/(1+e^y)*1/(1+e^x)
      Assert.assertEquals(message, 0.1192, gradientAb.get(1), 0.00001);
      Assert.assertTrue(message, VecTools.equals(gradientAb, gradientBa));
    }
  }

  @Test
  public void testSeqGradient2() {
    String message = "\n";
    final NFANetwork<Character> nfa = new NFANetwork<>(rng, 0., 3, 1, CharSeq.create("ab"));

    final NeuralSpider.NeuralNet ab = nfa.decisionByInput(CharSeq.create("ab"));
    final NeuralSpider.NeuralNet ba = nfa.decisionByInput(CharSeq.create("ba"));
    final Vec x = new ArrayVec(1, 0, 0, 1, 0, 1, 0, 0);
    message += nfa.ppState(ab.state(x), CharSeq.create("ab"));
    message += nfa.ppState(ba.state(x), CharSeq.create("ba"));

    final CompositeFunc target = new CompositeFunc(new WSum(new ArrayVec(0, 1)), new ParallelFunc(2, new Log(1., 0.)));
    final Vec gradientAb = ab.gradientTo(x, new ArrayVec(x.dim()), target);
    final Vec gradientBa = ba.gradientTo(x, new ArrayVec(x.dim()), target);
    message += "\nor: " + x + "\n"
        + "ab: " + gradientAb + "\n"
        + "ba: " + gradientBa + "\n";
    Assert.assertTrue(message, VecTools.equals(gradientAb, new Vec2CharSequenceConverter().convertFrom("8 -0.18654 -0.02541 0 0 -0.04347 -0.11817 -0.03956 -0.03956"), 0.00001));
    Assert.assertTrue(message, VecTools.equals(gradientBa, new Vec2CharSequenceConverter().convertFrom("8 -0.04167 -0.01533 -0.04167 -0.11327 -0.057 -0.15494 0 0"), 0.00001));
  }

  @Test
  public void testSeqGradient3() {
    String message = "\n";
    final NFANetwork<Character> nfa = new NFANetwork<>(rng, 0., 4, 1, CharSeq.create("ab"));

    final NeuralSpider.NeuralNet ab = nfa.decisionByInput(CharSeq.create("ab"));
    final NeuralSpider.NeuralNet ba = nfa.decisionByInput(CharSeq.create("ba"));
    final Vec x = new ArrayVec(
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
        0, 1, 0,
        0, 0, 1,
        0, 0, 0);
    message += nfa.ppState(ab.state(x), CharSeq.create("ab"));
    message += nfa.ppState(ba.state(x), CharSeq.create("ba"));
    final CompositeFunc target = new CompositeFunc(new WSum(new ArrayVec(0, 1)), new ParallelFunc(2, new Log(1., 0.)));

    final Vec gradientAb = ab.gradientTo(x, new ArrayVec(x.dim()), target);
    final Vec gradientBa = ba.gradientTo(x, new ArrayVec(x.dim()), target);
    message += "\nor: " + x + "\n"
        + "ab: " + gradientAb + "\n"
        + "ba: " + gradientBa + "\n";
    Assert.assertTrue(message, VecTools.equals(gradientAb, new Vec2CharSequenceConverter().convertFrom("18 -0.11105 -0.01512 -0.08577 0 0 0 0 0 0 -0.02588 -0.07035 -0.02588 -0.02355 -0.02355 -0.06401 0 0 0"), 0.00001));
  }

  @Test
  public void testSimpleSeq() {
    final int statesCount = 3;
    final CharSeq alpha = CharSeq.create("ab");
    final NFANetwork<Character> nfa = new NFANetwork<>(rng, 0.1, statesCount, 1, alpha);

    final PoolByRowsBuilder<FakeItem> pbuilder = new PoolByRowsBuilder<>(DataSetMeta.ItemType.FAKE);
    pbuilder.allocateFakeFeatures(1, FeatureMeta.ValueType.CHAR_SEQ);
    pbuilder.allocateFakeTarget(FeatureMeta.ValueType.INTS);
    pbuilder.setFeature(0, CharSeq.create("abba"));
    pbuilder.setTarget(0, 1);
    pbuilder.nextItem();
    pbuilder.setFeature(0, CharSeq.create("baba"));
    pbuilder.setTarget(0, 0);
    pbuilder.nextItem();
    final Pool<FakeItem> pool = pbuilder.create();

    final StochasticGradientDescent<FakeItem> gradientDescent = new StochasticGradientDescent<FakeItem>(rng, 1, 1000, 0.8) {
      @Override
      public void init(Vec cursor) {
        VecTools.fillUniform(cursor, rng);
      }
    };
    final Action<Vec> pp = new Action<Vec>() {
      int index = 0;

      @Override
      public void invoke(Vec vec) {
        if (++index == 1) {
          nfa.ppSolution(vec);
          {
            System.out.println("Positive: ");
            final NeuralSpider.NeuralNet abba = nfa.decisionByInput(CharSeq.create("abba"));
            System.out.println(nfa.ppState(abba.state(vec), CharSeq.create("abba")));
          }
          {
            System.out.println("Negative: ");
            final NeuralSpider.NeuralNet baba = nfa.decisionByInput(CharSeq.create("baba"));
            System.out.println(nfa.ppState(baba.state(vec), CharSeq.create("baba")));
          }
        }
      }
    };
    gradientDescent.addListener(pp);
    final BlockwiseMLL logit = pool.target(BlockwiseMLL.class);
    final DSSumFuncComposite<FakeItem> target = new DSSumFuncComposite<>(pool.data(), logit, argument -> {
      final CharSeq seq = pool.feature(0, argument.id);
      return nfa.decisionByInput(seq);
    });
    final DSSumFuncComposite<FakeItem>.Decision fit = gradientDescent.fit(pool.data(), target);

    final Vec solution = fit.x;
    nfa.ppSolution(solution);
    {
      System.out.println("Positive: ");
      final NeuralSpider.NeuralNet abba = nfa.decisionByInput(CharSeq.create("abba"));
      System.out.println(nfa.ppState(abba.state(solution), CharSeq.create("abba")));
      Assert.assertTrue(abba.trans(solution).get(1) > 0.95);
    }
    {
      System.out.println("Negative: ");
      final NeuralSpider.NeuralNet baba = nfa.decisionByInput(CharSeq.create("baba"));
      System.out.println(nfa.ppState(baba.state(solution), CharSeq.create("baba")));
      Assert.assertTrue(baba.trans(solution).get(1) < 0.05);
    }
  }

  @Test
  public void testUrlConvergence() throws Exception {
    //noinspection unchecked
    final PoolByRowsBuilder<FakeItem> pbuilder = new PoolByRowsBuilder<>(DataSetMeta.ItemType.FAKE);
    pbuilder.allocateFakeFeatures(1, FeatureMeta.ValueType.CHAR_SEQ);
    pbuilder.allocateFakeTarget(FeatureMeta.ValueType.INTS);
    final Set<Character> alphaSet = new HashSet<>();
    boolean next = true;
    for (int i = 0; i < featuresTxtPool.data().length(); i++) {
      final QURLItem item = featuresTxtPool.data().at(i);
      final int seqClass = item.url.substring("https://".length(), item.url.length() - 1).contains("/") ? 1 : 0;
      if (seqClass > 0 != next)
        continue;
      next = !next;
      final Seq<Character> url = CharSeq.create(item.url.substring("http://".length()));
      pbuilder.setFeature(0, url);
      pbuilder.setTarget(0, seqClass);
      for (int j = 0; j < url.length(); j++) {
        alphaSet.add(url.at(j));
      }
      pbuilder.nextItem();
    }

    final Pool<FakeItem> pool = pbuilder.create();

    final CharSeqArray alpha = new CharSeqArray(alphaSet.toArray(new Character[alphaSet.size()]));
    final int statesCount = 4;
    final NFANetwork<Character> network = new NFANetwork<>(rng, 0.1, statesCount, 1, alpha);

    final StochasticGradientDescent<FakeItem> gradientDescent = new StochasticGradientDescent<FakeItem>(rng, 4, 2000, 0.8) {
      @Override
      public void init(Vec cursor) {
        final int paramsDim = (statesCount - 1) * (statesCount - 1);
        for (int i = 0; i < alpha.length(); i++) {
          final VecBasedMx mx = new VecBasedMx(statesCount - 1, cursor.sub(i * paramsDim, paramsDim));
          for (int j = 0; j < mx.rows(); j++) {
            mx.set(j, j, 5);
          }
        }
      }
    };
    final BlockwiseMLL ll = pool.target(BlockwiseMLL.class);
    final Action<Vec> pp = new Action<Vec>() {
      int index = 0;

      @Override
      public void invoke(Vec vec) {
        if (++index % 100 == 1) {
          network.ppSolution(vec, '/');
        }
      }
    };
    gradientDescent.addListener(pp);
    final DSSumFuncComposite<FakeItem> target = new DSSumFuncComposite<>(pool.data(), ll, argument -> {
      final CharSeq seq = pool.feature(0, argument.id);
      return network.decisionByInput(seq);
    });
    final DSSumFuncComposite<FakeItem>.Decision fit = gradientDescent.fit(pool.data(), target);
    final Vec solution = fit.x;
    digIntoSolution(pool, network, ll, solution, "www.yandex.ru/yandsearch?text=xyu", "www.yandex.ru");
  }

  @Test
  public void testUrlConvergence2() throws Exception {
    //noinspection unchecked
    final PoolByRowsBuilder<FakeItem> pbuilder = new PoolByRowsBuilder<>(DataSetMeta.ItemType.FAKE);
    pbuilder.allocateFakeFeatures(1, FeatureMeta.ValueType.CHAR_SEQ);
    pbuilder.allocateFakeTarget(FeatureMeta.ValueType.INTS);
    final Set<Character> alphaSet = new HashSet<>();
    boolean next = true;
    for (int i = 0; i < featuresTxtPool.data().length(); i++) {
      final QURLItem item = featuresTxtPool.data().at(i);
      final int seqClass = item.url.substring("https://".length()).contains("htm") ? 1 : 0;
      if (seqClass > 0 != next)
        continue;
      next = !next;
      final Seq<Character> url = CharSeq.create(item.url.substring("http://".length()));
      pbuilder.setFeature(0, url);
      pbuilder.setTarget(0, seqClass);
      for (int j = 0; j < url.length(); j++) {
        alphaSet.add(url.at(j));
      }
      pbuilder.nextItem();
    }

    final Pool<FakeItem> pool = pbuilder.create();

    final CharSeqArray alpha = new CharSeqArray(alphaSet.toArray(new Character[alphaSet.size()]));
    final int statesCount = 5;
    final int finalStates = 1;
    final NFANetwork<Character> network = new NFANetwork<>(rng, 0.1, statesCount, finalStates, alpha);
    final StochasticGradientDescent<FakeItem> gradientDescent = getStochasticGradientDescent(alpha, statesCount,
            finalStates, 10, 20000, 2);
    final Action<Vec> pp = new Action<Vec>() {
      int index = 0;

      @Override
      public void invoke(Vec vec) {
        if (++index % 1000 == 1) {
          network.ppSolution(vec, 'h');
          network.ppSolution(vec, 't');
          network.ppSolution(vec, 'm');
          network.ppSolution(vec, 'l');
          network.ppSolution(vec, '.');
        }
      }
    };
    gradientDescent.addListener(pp);
    final BlockwiseMLL ll = pool.target(BlockwiseMLL.class);
    final ArrayVec initial = new ArrayVec(network.dim());
    gradientDescent.init(initial);
//    digIntoSolution(pool, network, ll, initial, "www.yandex.ru/yandsearch?text=xyu.htm", "www.yandex.ru");

    final DSSumFuncComposite<FakeItem> target = new DSSumFuncComposite<>(pool.data(), ll, argument -> {
      final CharSeq seq = pool.feature(0, argument.id);
      return network.decisionByInput(seq);
    });
    final DSSumFuncComposite<FakeItem>.Decision fit = gradientDescent.fit(pool.data(), target);
//    digIntoSolution(pool, network, ll, fit.x, "www.yamdex.ru/yandsearch?text=xyu.htm", "www.yamdex.ru");
    digIntoSolutionParallel(pool, network, ll, fit);
  }

  @Test
  public void testSeqConvergence() throws Exception {
    final PoolByRowsBuilder<FakeItem> pbuilder = new PoolByRowsBuilder<>(DataSetMeta.ItemType.FAKE);
    pbuilder.allocateFakeFeatures(1, FeatureMeta.ValueType.CHAR_SEQ);
    pbuilder.allocateFakeTarget(FeatureMeta.ValueType.INTS);
    CharSeqTools.processLines(
        new InputStreamReader(new GZIPInputStream(new FileInputStream("src/test/resources/com/spbsu/ml/train.txt.gz"))),
        new Processor<CharSequence>() {
          CharSequence[] parts = new CharSequence[2];
          boolean next = true;
          int index = 0;

          @Override
          public void process(CharSequence arg) {
            if (++index % 100 == 1) {
              CharSeqTools.split(arg, '\t', parts);
              final CharSeq next = CharSeq.create(parts[0]);
              final int nextClass = CharSeqTools.parseInt(CharSeqTools.split(parts[1], ':')[1]);
              if (nextClass > 0 != this.next || next.length() > 20)
                return;
              this.next = !this.next;
              pbuilder.setFeature(0, next);
              pbuilder.setTarget(0, nextClass);
              pbuilder.nextItem();
            }
          }
        });

    final Pool<FakeItem> pool = pbuilder.create();
    System.out.println(pool.data().length());
    final CharSeqArray alpha = new CharSeqArray('U', 'L', 'H', 'C', 'S', 'N', 'R', 'F', 'V', 'O');
    final int statesCount = 10;
    final NFANetwork<Character> network = new NFANetwork<>(rng, 0.5, statesCount, 1, alpha);
    final StochasticGradientDescent<FakeItem> gradientDescent = new StochasticGradientDescent<FakeItem>(rng, 10, 1000, 1) {
      @Override
      public void init(Vec cursor) {
        final int paramsDim = (statesCount - 1) * (statesCount - 1);
        initCursor(cursor, paramsDim, alpha, statesCount);
      }

      @Override
      public void normalizeGradient(Vec grad) {
//        for (int i = 0; i < grad.length(); i++) {
//          final double v = grad.get(i);
//          if (Math.abs(v) < 0.001)
//            grad.set(i, 0);
//          else
//            grad.set(i, Math.signum(v) * (Math.abs(v) - 0.001));
//        }
      }
    };
    final BlockwiseMLL ll = pool.target(BlockwiseMLL.class);
    final Action<Vec> pp = new Action<Vec>() {
      int index = 0;

      @Override
      public void invoke(Vec vec) {
        if (++index % 10 == 1) {
          double sum = 0;
          int count = 0;
          int negative = 0;
          for (int i = 0; i < 1000; i++, count++) {
            final double value = ll.block(i).value(network.decisionByInput(pool.feature(0, i)).trans(vec));
            sum += value;
            if (Math.exp(-value) > 2)
              negative++;
          }
          System.out.println(index + " ll: " + Math.exp(-sum / count) + " prec: " + (count - negative) / (double) count);
        }
      }
    };
    gradientDescent.addListener(pp);
    final DSSumFuncComposite<FakeItem> target = new DSSumFuncComposite<>(pool.data(), ll, argument -> {
      final CharSeq seq = pool.feature(0, argument.id);
      return network.decisionByInput(seq);
    });
    final DSSumFuncComposite<FakeItem>.Decision fit = gradientDescent.fit(pool.data(), target);
    final Vec vals = new ArrayVec(pool.size());
    for (int i = 0; i < vals.length(); i++) {
      vals.set(i, fit.compute(pool.data().at(i)).get(0));
    }
    network.ppSolution(fit.x);
    digIntoSolutionParallel(pool, network, ll, fit);
    System.out.println(Math.exp(-ll.value(vals) / ll.dim()));
  }

  @Test
  public void testSimpleMLLSeq() {
    final PoolByRowsBuilder<FakeItem> pbuilder = new PoolByRowsBuilder<>(DataSetMeta.ItemType.FAKE);
    pbuilder.allocateFakeFeatures(1, FeatureMeta.ValueType.CHAR_SEQ);
    pbuilder.allocateFakeTarget(FeatureMeta.ValueType.INTS);
    final CharSequence[] secs = {"abrakadabra", "balalayka", "infinity", "element"};
    final Set<Character> alphaSet = new HashSet<>();
    final Pool<FakeItem> pool = getFakeItemPool(pbuilder, secs, alphaSet);

    final CharSeqArray alpha = new CharSeqArray(alphaSet.toArray(new Character[alphaSet.size()]));
    final int statesCount = 8;
    final int finalStates = 4;
    final NFANetwork<Character> network = new NFANetwork<>(rng, 0.1, statesCount, finalStates, alpha);
    final StochasticGradientDescent<FakeItem> gradientDescent = getStochasticGradientDescent(alpha, statesCount,
            finalStates, 10, 20000, 2);

    final Action<Vec> pp = getVecAction(secs, network);
    gradientDescent.addListener(pp);

    final BlockwiseMLL ll = pool.target(BlockwiseMLL.class);
    final ArrayVec initial = new ArrayVec(network.dim());
    gradientDescent.init(initial);
    final DSSumFuncComposite<FakeItem> target = new DSSumFuncComposite<>(pool.data(), ll, argument -> {
      final CharSeq seq = pool.feature(0, argument.id);
      return network.decisionByInput(seq);
    });
    final DSSumFuncComposite<FakeItem>.Decision decision = gradientDescent.fit(pool.data(), target);
    System.out.println(decision.x);

    for (int i = 0; i < secs.length; i++) {
      System.out.println("Class: " + i);
      System.out.println(decision.compute(pool.data().at(i)));
    }
  }


  @Test
  public void testGenom() throws IOException {
    final Set<Character> alphaSet = new HashSet<>();
    final Map<String, List<Seq<Character>>> map = new HashMap<>();

    final List<String> types = new ArrayList<>();
    final File file = new File("src/test/resources/com/spbsu/ml/multiclass/HiSeq.txt");
    final BufferedReader br = new BufferedReader(new FileReader(file));
    String line;
    while ((line = br.readLine()) != null) {
      String[] strs = line.split("->");
      if (strs.length < 2)
        continue;
      String name = strs[0];
      String value = strs[1];
      final Seq<Character> genom = CharSeq.create(value);
      if (genom.length() == 101) {
        if (!map.keySet().contains(name))
          map.put(name, new ArrayList<>());
        if (!types.contains(name))
          types.add(name);
        for (int j = 0; j < genom.length(); j++)
          alphaSet.add(genom.at(j));
        map.get(name).add(genom);
      }
    }
    br.close();


    final PoolByRowsBuilder<FakeItem> pbuilder = new PoolByRowsBuilder<>(DataSetMeta.ItemType.FAKE);
    pbuilder.allocateFakeFeatures(1, FeatureMeta.ValueType.CHAR_SEQ);
    pbuilder.allocateFakeTarget(FeatureMeta.ValueType.INTS);
    for (String name : types) {
      for (Seq<Character> genom : map.get(name)) {
        pbuilder.setFeature(0, genom);
        pbuilder.setTarget(0, types.indexOf(name) + 1);
        pbuilder.nextItem();
      }
    }
    final Pool<FakeItem> pool = pbuilder.create();


    for (int i = 0; i < types.size(); i++) {
      String name = types.get(i);
      System.out.println(String.format("%s (%s) -> %s", name, i + 1, map.get(name).size()));
    }
    System.out.println(alphaSet);

    final CharSeqArray alpha = new CharSeqArray(alphaSet.toArray(new Character[alphaSet.size()]));
    final int statesCount = 13;
    final int finalStates = types.size();
    final NFANetwork<Character> network = new NFANetwork<>(rng, 0.1, statesCount, finalStates, alpha);
    int iterations = 1000;
    final StochasticGradientDescent<FakeItem> gradientDescent = getStochasticGradientDescent(alpha, statesCount, finalStates,
            4, iterations, 2);

    final Action<Vec> pp = new Action<Vec>() {
      double index = 0;

      @Override
      public void invoke(Vec vec) {
        index++;
        if (index % 50 == 0) {
          System.out.println("iterations: " + index + ", " + index / iterations * 100 + "%");
        }
      }
    };
    gradientDescent.addListener(pp);

    final BlockwiseMLL ll = pool.target(BlockwiseMLL.class);
    final ArrayVec initial = new ArrayVec(network.dim());
    gradientDescent.init(initial);

    final DSSumFuncComposite<FakeItem> target = new DSSumFuncComposite<>(pool.data(), ll, argument -> {
      final CharSeq seq = pool.feature(0, argument.id);
      return network.decisionByInput(seq);
    });
    final DSSumFuncComposite<FakeItem>.Decision decision = gradientDescent.fit(pool.data(), target);
    System.out.println();

    digIntoSolutionParallel(pool, network, ll, decision);
  }


  @Test
  public void weightsCalculatorTest() {
    Vec betta = new ArrayVec(36);
    VecTools.fillUniform(betta, rng);
    boolean[] booleans = {false, false, false, false, false};

    WeightsCalculator weightsCalculator1 = new WeightsCalculator(5, 2, 12, 12);
    weightsCalculator1.setDropOut(booleans);
    Mx mx1 = weightsCalculator1.compute(betta);

    WeightsCalculator weightsCalculator2 = new WeightsCalculator(5, 2, 0, 12);
    weightsCalculator2.setDropOut(booleans);
    Mx mx2 = weightsCalculator2.compute(betta);

    WeightsCalculator weightsCalculator3 = new WeightsCalculator(5, 2, 24, 12);
    weightsCalculator3.setDropOut(booleans);
    Mx mx3 = weightsCalculator3.compute(betta);

    SeqWeightsCalculator weightsCalculator = new SeqWeightsCalculator(5, 2, 12, 12, 0, 24);
    weightsCalculator.setDropOut(booleans);
    Mx mx = weightsCalculator.compute(betta);

    Mx multiply = MxTools.multiply(mx3, MxTools.multiply(mx2, mx1));

    for (int i = 0; i < mx.rows(); i++) {
      for (int j = 0; j < mx.columns(); j++) {
        Assert.assertEquals(multiply.get(i, j), mx.get(i, j), 0.0001);
      }
    }
  }


  @Test
  public void testSimpleMLLSeqWithLZW() {
    final PoolByRowsBuilder<FakeItem> pbuilder = new PoolByRowsBuilder<>(DataSetMeta.ItemType.FAKE);
    pbuilder.allocateFakeFeatures(1, FeatureMeta.ValueType.CHAR_SEQ);
    pbuilder.allocateFakeTarget(FeatureMeta.ValueType.INTS);
    final CharSequence[] secs = {"abrakadabra", "balalayka", "infinity", "element"};
    final Set<Character> alphaSet = new HashSet<>();
    final Pool<FakeItem> pool = getFakeItemPool(pbuilder, secs, alphaSet);

    final CharSeqArray alpha = new CharSeqArray(alphaSet.toArray(new Character[alphaSet.size()]));
    final int statesCount = 8;
    final int finalStates = 4;
    final NFANetworkWithLZW<Character> network = new NFANetworkWithLZW<>(rng, 0.1, statesCount, finalStates, alpha);
    final StochasticGradientDescent<FakeItem> gradientDescent = getStochasticGradientDescent(alpha, statesCount,
            finalStates, 10, 20000, 2);

    final Action<Vec> pp = getVecAction(secs, network);
    gradientDescent.addListener(pp);

    final BlockwiseMLL ll = pool.target(BlockwiseMLL.class);
    final ArrayVec initial = new ArrayVec(network.dim());
    gradientDescent.init(initial);
    final DSSumFuncComposite<FakeItem> target = new DSSumFuncComposite<>(pool.data(), ll, argument -> {
      final CharSeq seq = pool.feature(0, argument.id);
      return network.decisionByInput(seq);
    });
    final DSSumFuncComposite<FakeItem>.Decision decision = gradientDescent.fit(pool.data(), target);
    System.out.println(decision.x);

    for (int i = 0; i < secs.length; i++) {
      System.out.println("Class: " + i);
      System.out.println(decision.compute(pool.data().at(i)));
    }
  }


  @Test
  public void weightsCalculatorGradTest() {
    Vec betta = new ArrayVec(24);
    VecTools.fillUniform(betta, rng);
    boolean[] booleans = {false, false, false, false, false};

    System.out.println("betta: " + betta);

    Vec x = new ArrayVec(5);
    VecTools.fillUniform(x, rng);
    System.out.println("x: " + x);

    WeightsCalculator weightsCalculator1 = new WeightsCalculator(5, 2, 0, 12);
    weightsCalculator1.setDropOut(booleans);
    Mx mx1 = weightsCalculator1.compute(betta);
    Vec state1 = MxTools.multiply(mx1, x);

    System.out.println("mx1:" + mx1);
    System.out.println("state1: " + state1);

    WeightsCalculator weightsCalculator2 = new WeightsCalculator(5, 2, 12, 12);
    weightsCalculator2.setDropOut(booleans);
    Mx mx2 = weightsCalculator2.compute(betta);
    Vec state2 = MxTools.multiply(mx2, state1);

    System.out.println("mx2:" + mx2);
    System.out.println("state2: " + state2);

    Vec states = new ArrayVec(15);
    for (int i = 0; i < x.length(); i++) {
      states.set(i, x.get(i));
    }
    for (int i = 0; i < state1.length(); i++) {
      states.set(5 + i, state1.get(i));
    }
    for (int i = 0; i < state2.length(); i++) {
      states.set(10 + i, state2.get(i));
    }

    System.out.println("states: " + states);

    Vec grad = new ArrayVec(24);
    final int indexOfNode = rng.nextInt(5);
    System.out.println("indexOfNode: " + indexOfNode);

    NFANetwork.MyNode node2 = new NFANetwork.MyNode(indexOfNode, 12, 12, 24, 5, 5, 15, weightsCalculator2);
    node2.transByParents(states).gradientTo(betta, grad);
    System.out.println("grad: " + grad);

    for (int i = 0; i < 3; i++) {
      Vec curGrad = new ArrayVec(24);
      NFANetwork.MyNode node1 = new NFANetwork.MyNode(i, 0, 12, 24, 0, 5, 15, weightsCalculator1);
      node1.transByParents(states).gradientTo(betta, curGrad);
      VecTools.scale(curGrad, mx2.get(indexOfNode, i));
      VecTools.append(grad, curGrad);
    }

    System.out.println("grad: " + grad);

    Vec seqGrad = new ArrayVec(24);
    SeqWeightsCalculator weightsCalculator = new SeqWeightsCalculator(5, 2, 12, 0, 12);
    weightsCalculator.setDropOut(booleans);
    weightsCalculator.gradientTo(betta, seqGrad, x, indexOfNode);

    System.out.println("seqGrad: " + seqGrad);

    for (int i = 0; i < grad.length(); i++) {
      Assert.assertEquals(grad.get(i), seqGrad.get(i), 0.0001);
    }
  }


  private void digIntoSolution(Pool<FakeItem> pool, NFANetwork<Character> network, BlockwiseMLL ll, Vec solution, String positiveExample, String negativeExample) {
    if (positiveExample != null) {
      System.out.println("Positive: ");
      final CharSeq input = CharSeq.create(positiveExample);
      final NeuralSpider.NeuralNet positive = network.decisionByInput(input);
      System.out.println(network.ppState(positive.state(solution), input));
    }
    if (negativeExample != null) {
      System.out.println("Negative: ");
      final CharSeq input = CharSeq.create(negativeExample);
      final NeuralSpider.NeuralNet negative = network.decisionByInput(input);
      System.out.println(network.ppState(negative.state(solution), input));
    }

    int count = 0, negative = 0;
    double llSum = 0;
    for (int i = 0; i < ll.blocksCount(); i++) {
      final CharSeq input = pool.feature(0, i);
      final double llblock = ll.block(i).value(network.decisionByInput(input).trans(solution));
      llSum += llblock;
      final double pX = Math.exp(llblock);
      count++;
      if (pX < 0.5) {
        negative++;
        System.out.println("Input: [" + input + "]");
        final NeuralSpider.NeuralNet net = network.decisionByInput(input);
        System.out.println(network.ppState(net.state(solution), input));
        System.out.println();
      }
    }
    System.out.println(Math.exp(-llSum / ll.dim()) + " " + (count - negative) / (double) count);
    Assert.assertTrue(1.1 > Math.exp(-llSum / ll.dim()));
  }

  private void digIntoSolutionParallel(Pool<FakeItem> pool, NFANetwork<Character> network, BlockwiseMLL ll, DSSumFuncComposite<FakeItem>.Decision decision) {
    try {
      ExecutorService executorService = Executors.newFixedThreadPool(4);
      List<Callable<Vec>> tasks = new ArrayList<>(ll.blocksCount());
      for (int i = 0; i < ll.blocksCount(); i++) {
        final int finalI = i;
        //tasks.add(() -> ll.block(finalI).value(network.decisionByInput(pool.feature(0, finalI)).trans(decision.x)));
        tasks.add(() -> network.decisionByInput(pool.feature(0, finalI)).trans(decision.x));
      }
      List<Future<Vec>> list = new ArrayList<>(tasks.size());
      for (int i = 0; i < tasks.size(); i++) {
        list.add(executorService.submit(tasks.get(i)));
      }

      List<Integer> finishedTasks = new ArrayList<>();
      int count = 0, negative = 0, label = -1;
      Map<Integer, Integer> positives = new HashMap<>();
      double llSum = 0;
      while (count < ll.blocksCount()) {
        Thread.sleep(1000);
        for (int i = 0; i < ll.blocksCount(); i++) {
          if (list.get(i).isDone() && !finishedTasks.contains(i)) {
            finishedTasks.add(i);
            Vec vec = list.get(i).get();
            double llblock = ll.block(i).value(vec);
            int rightAnswer = ll.label(i);
            if (VecTools.argmax(vec) == rightAnswer) {
              Integer integer = positives.get(rightAnswer);
              if (integer == null) {
                positives.put(rightAnswer, 1);
              } else {
                positives.put(rightAnswer, integer + 1);
              }
            }
            llSum += llblock;
            final double pX = Math.exp(llblock);
            count++;
            System.out.print(String.format("Calculating results: %s (%.2f%%)\r", count, (double) count / ll.blocksCount() * 100));
            if (pX < 1.0 / ll.classesCount()) {
              negative++;
              if (label != rightAnswer) {
                label = rightAnswer;
                CharSeq input = pool.feature(0, i);
                System.out.println("Input: [" + input + "]");
                System.out.println("Output: [" + rightAnswer + "]");
                final NeuralSpider.NeuralNet net = network.decisionByInput(input);
                System.out.println(network.ppState(net.state(decision.x), input));
                System.out.println();
              }
            }
          }
        }
      }
      System.out.println(ll.transformResultValue(llSum) + " " + (count - negative) / (double) count);
      System.out.println(positives);
      Assert.assertTrue(1.1 > ll.transformResultValue(llSum));
    } catch (InterruptedException | ExecutionException e) {
      throw new RuntimeException(e);
    }
  }

  private Pool<FakeItem> getFakeItemPool(PoolByRowsBuilder<FakeItem> pbuilder, CharSequence[] secs, Set<Character> alphaSet) {
    for (int i = 0; i < secs.length; i++) {
      final Seq<Character> url = CharSeq.create(secs[i]);
      pbuilder.setFeature(0, url);
      pbuilder.setTarget(0, i + 1);
      for (int j = 0; j < url.length(); j++) {
        alphaSet.add(url.at(j));
      }
      pbuilder.nextItem();
    }
    return pbuilder.create();
  }


  @NotNull
  private StochasticGradientDescent<FakeItem> getStochasticGradientDescent(final CharSeqArray alpha, final int statesCount,
                                                                           final int finalStates, int couple, int T, double step) {
    return new StochasticGradientDescent<FakeItem>(rng, couple, T, step) {
      @Override
      public void init(Vec cursor) {
        final int paramsDim = (statesCount - finalStates) * (statesCount - 1);
        initCursor(cursor, paramsDim, alpha, statesCount);
      }

      @Override
      public void normalizeGradient(Vec grad) {
        for (int i = 0; i < grad.length(); i++) {
          if (Math.abs(grad.get(i)) < 0.001)
            grad.set(i, 0);
        }
      }
    };
  }

  private void initCursor(Vec cursor, int paramsDim, CharSeqArray alpha, int statesCount) {
    for (int c = 0; c < alpha.length(); c++) {
      final VecBasedMx mx = new VecBasedMx(statesCount - 1, cursor.sub(c * paramsDim, paramsDim));
      VecTools.fillUniform(mx, rng, 5. / (statesCount - 1));
      for (int j = 0; j < mx.rows(); j++) {
        mx.set(j, j, 5);
      }
    }
  }


  @NotNull
  private Action<Vec> getVecAction(final CharSequence[] secs, final NeuralSpider<Character, Seq<Character>> network) {
    return new Action<Vec>() {
      int index = 0;

      @Override
      public void invoke(Vec vec) {
        index++;
        if (index % 10000 == 1) {
          for (int i = 0; i < secs.length; i++) {
            System.out.println("Class: " + i);
            final NeuralSpider.NeuralNet neuralNet = network.decisionByInput(CharSeq.create(secs[i]));
            System.out.println(network.ppState(neuralNet.state(vec), CharSeq.create(secs[i])));
          }
        }
      }
    };
  }

}
