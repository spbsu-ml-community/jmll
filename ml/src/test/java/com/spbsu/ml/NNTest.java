package com.spbsu.ml;

import com.spbsu.commons.func.Action;
import com.spbsu.commons.func.Computable;
import com.spbsu.commons.func.Processor;
import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.math.Trans;
import com.spbsu.commons.math.TransC1;
import com.spbsu.commons.math.io.Vec2CharSequenceConverter;
import com.spbsu.commons.math.vectors.Mx;
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
import com.spbsu.ml.loss.MLL;
import com.spbsu.ml.meta.DataSetMeta;
import com.spbsu.ml.meta.FeatureMeta;
import com.spbsu.ml.meta.items.FakeItem;
import com.spbsu.ml.meta.items.QURLItem;
import com.spbsu.ml.methods.StochasticGradientDescent;
import com.spbsu.ml.models.nn.NeuralSpider;
import com.spbsu.ml.models.nn.LayeredNetwork;
import com.spbsu.ml.models.nn.nfa.NFANetwork;
import com.spbsu.ml.testUtils.TestResourceLoader;
import org.junit.Assert;
import org.junit.Test;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashSet;
import java.util.Set;
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
    final DSSumFuncComposite<FakeItem> target = new DSSumFuncComposite<>(pool.data(), ll, new Computable<FakeItem, TransC1>() {
      @Override
      public NeuralSpider.NeuralNet compute(final FakeItem argument) {
        final Vec row = data.row(argument.id);
        return network.decisionByInput(row);
      }
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
      final NFANetwork<Character> nfa = new NFANetwork<>(rng, 0., 5, CharSeq.create("ab"));
      final Trans aba = nfa.decisionByInput(CharSeq.create("aba"));
      Assert.assertEquals(0.2 + 0.16 + 0.128, aba.trans(new ArrayVec(nfa.dim())).get(1), 0.0001);
  }

  @Test
  public void testSeqGradient1() {
    String message = "\n";

    final NFANetwork<Character> nfa = new NFANetwork<>(rng, 0., 2, CharSeq.create("ab"));

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

    final NFANetwork<Character> nfa = new NFANetwork<>(rng, 0., 3, CharSeq.create("ab"));

    final NeuralSpider.NeuralNet ab = nfa.decisionByInput(CharSeq.create("ab"));
    final NeuralSpider.NeuralNet ba = nfa.decisionByInput(CharSeq.create("ba"));
    final Vec x = new ArrayVec(1,0, 0,1,  0,1, 0,0);
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

    final NFANetwork<Character> nfa = new NFANetwork<>(rng, 0., 4, CharSeq.create("ab"));

    final NeuralSpider.NeuralNet ab = nfa.decisionByInput(CharSeq.create("ab"));
    final NeuralSpider.NeuralNet ba = nfa.decisionByInput(CharSeq.create("ba"));
    final Vec x = new ArrayVec(
            1,0,0,
            0,1,0,
            0,0,1,

            0,1,0,
            0,0,1,
            0,0,0);
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
    final NFANetwork<Character> nfa = new NFANetwork<>(rng, 0.1, statesCount, alpha);

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
        final int paramsDim = (statesCount - 1) * (statesCount - 1);
        VecTools.fillUniform(cursor, rng);
//        for (int i = 0; i < alpha.length(); i++) {
//          final VecBasedMx mx = new VecBasedMx(statesCount - 1, cursor.sub(i * paramsDim, paramsDim));
//          for (int j = 0; j < mx.rows(); j++) {
//            mx.set(j, j, statesCount);
//          }
//        }
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
    final MLL logit = pool.target(MLL.class);
    final DSSumFuncComposite<FakeItem> target = new DSSumFuncComposite<>(pool.data(), logit, new Computable<FakeItem, TransC1>() {
      @Override
      public TransC1 compute(final FakeItem argument) {
        final CharSeq seq = pool.feature(0, argument.id);
        return nfa.decisionByInput(seq);
      }
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
    final NFANetwork<Character> network = new NFANetwork<>(rng, 0.1, statesCount, alpha);

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
    final MLL ll = pool.target(MLL.class);
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
    final DSSumFuncComposite<FakeItem> target = new DSSumFuncComposite<>(pool.data(), ll, new Computable<FakeItem, TransC1>() {
      @Override
      public NeuralSpider.NeuralNet compute(final FakeItem argument) {
        final CharSeq seq = pool.feature(0, argument.id);
        return network.decisionByInput(seq);
      }
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
    final NFANetwork<Character> network = new NFANetwork<>(rng, 0.1, statesCount, alpha);
    final StochasticGradientDescent<FakeItem> gradientDescent = new StochasticGradientDescent<FakeItem>(rng, 10, 20000, 2){
      @Override
      public void init(Vec cursor) {
        final int paramsDim = (statesCount - 1) * (statesCount - (NFANetwork.OUTPUT_NODES - 1));
        for (int c = 0; c < alpha.length(); c++) {
          final VecBasedMx mx = new VecBasedMx(statesCount - 1, cursor.sub(c * paramsDim, paramsDim));
          VecTools.fillUniform(mx, rng, 5. / (statesCount - 1));
          for (int j = 0; j < mx.rows(); j++) {
            mx.set(j, j, 5);
          }
        }
      }

      @Override
      public void normalizeGradient(Vec grad) {
        for (int i = 0; i < grad.length(); i++) {
          if (Math.abs(grad.get(i)) < 0.001)
            grad.set(i, 0);
        }
      }
    };
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
    final MLL ll = pool.target(MLL.class);
    final ArrayVec initial = new ArrayVec(network.dim());
    gradientDescent.init(initial);
//    digIntoSolution(pool, network, ll, initial, "www.yandex.ru/yandsearch?text=xyu.htm", "www.yandex.ru");

    final DSSumFuncComposite<FakeItem> target = new DSSumFuncComposite<>(pool.data(), ll, new Computable<FakeItem, TransC1>() {
      @Override
      public NeuralSpider.NeuralNet compute(final FakeItem argument) {
        final CharSeq seq = pool.feature(0, argument.id);
        return network.decisionByInput(seq);
      }
    });
    final DSSumFuncComposite<FakeItem>.Decision fit = gradientDescent.fit(pool.data(), target);
    digIntoSolution(pool, network, ll, fit.x, "www.yamdex.ru/yandsearch?text=xyu.htm", "www.yamdex.ru");
  }

  @Test
  public void testSeqConvergence() throws Exception {
    final PoolByRowsBuilder<FakeItem> pbuilder = new PoolByRowsBuilder<>(DataSetMeta.ItemType.FAKE);
    pbuilder.allocateFakeFeatures(1, FeatureMeta.ValueType.CHAR_SEQ);
    pbuilder.allocateFakeTarget(FeatureMeta.ValueType.INTS);
    CharSeqTools.processLines(
            new InputStreamReader(new GZIPInputStream(new FileInputStream("/Users/solar/tree/java/relpred/trunk/relpred/main/tests/data/in/train.txt.gz"))),
            new Processor<CharSequence>() {
              CharSequence[] parts = new CharSequence[2];
              boolean next = true;

              @Override
              public void process(CharSequence arg) {
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
            });

    final Pool<FakeItem> pool = pbuilder.create();
    final CharSeqArray alpha = new CharSeqArray('U', 'L', 'H', 'C', 'S', 'N', 'R', 'F', 'V', 'O');
    final int statesCount = 10;
    final NFANetwork<Character> network = new NFANetwork<>(rng, 0.5, statesCount, alpha);
    final StochasticGradientDescent<FakeItem> gradientDescent = new StochasticGradientDescent<FakeItem>(rng, 4, 1000000, 1) {
      @Override
      public void init(Vec cursor) {
        final int paramsDim = (statesCount - 1) * (statesCount - 1);
        for (int c = 0; c < alpha.length(); c++) {
          final VecBasedMx mx = new VecBasedMx(statesCount - 1, cursor.sub(c * paramsDim, paramsDim));
          VecTools.fillUniform(mx, rng, 5. / (statesCount - 1));
          for (int j = 0; j < mx.rows(); j++) {
            mx.set(j, j, 5);
          }
        }
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
    final MLL ll = pool.target(MLL.class);
    final Action<Vec> pp = new Action<Vec>() {
      int index = 0;
      @Override
      public void invoke(Vec vec) {
        if (++index % 10 == 1) {
          double sum = 0;
          int count = 0;
          int negative = 0;
          for (int i = 0; i < 1000; i++, count++) {
            final double value = ll.block(i).value(network.decisionByInput((CharSeq) pool.feature(0, i)).trans(vec));
            sum += value;
            if (Math.exp(-value) > 2)
              negative++;
          }
          System.out.println(index + " ll: " + Math.exp(-sum / count) + " prec: " + (count - negative)/(double)count);
        }
      }
    };
    gradientDescent.addListener(pp);
    final DSSumFuncComposite<FakeItem> target = new DSSumFuncComposite<>(pool.data(), ll, new Computable<FakeItem, TransC1>() {
      @Override
      public NeuralSpider.NeuralNet compute(final FakeItem argument) {
        final CharSeq seq = pool.feature(0, argument.id);
        return network.decisionByInput(seq);
      }
    });
    final DSSumFuncComposite<FakeItem>.Decision fit = gradientDescent.fit(pool.data(), target);
    final Vec vals = new ArrayVec(pool.size());
    for (int i = 0; i < vals.length(); i++) {
      vals.set(i, fit.compute(pool.data().at(i)).get(0));
    }
    network.ppSolution(fit.x);
    digIntoSolution(pool, network, ll, fit.x, null, null);
    System.out.println(Math.exp(-ll.value(vals) / ll.dim()));
  }


  private void digIntoSolution(Pool<FakeItem> pool, NFANetwork<Character> network, MLL ll, Vec solution, String positiveExample, String negativeExample) {
    if (positiveExample != null) {
      System.out.println("Positive: ");
      final CharSeq input = CharSeq.create(positiveExample);
      final NeuralSpider.NeuralNet positive = network.decisionByInput(input);
      System.out.println(network.ppState(positive.state(solution), input));
//      Vec gradient = positive.gradient(solution);
//      network.ppSolution(gradient, 'h');
//      network.ppSolution(gradient, 't');
//      network.ppSolution(gradient, 'm');
//      network.ppSolution(gradient, 'l');
//      network.ppSolution(gradient, '.');
//      Assert.assertTrue(positive.value(solution) > 0.95);
    }
    if (negativeExample != null) {
      System.out.println("Negative: ");
      final CharSeq input = CharSeq.create(negativeExample);
      final NeuralSpider.NeuralNet negative = network.decisionByInput(input);
      System.out.println(network.ppState(negative.state(solution), input));
//      network.ppSolution(negative.gradient(solution), 'h');
//      network.ppSolution(negative.gradient(solution), 't');
//      network.ppSolution(negative.gradient(solution), 'm');
//      network.ppSolution(negative.gradient(solution), 'l');
//      network.ppSolution(negative.gradient(solution), '.');
//      Assert.assertTrue(negative.value(solution) < 0.05);
    }

//    network.ppState(fit.x);
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
    System.out.println(Math.exp(-llSum / ll.blocksCount()) + " " + (count - negative) / (double)count);
    Assert.assertTrue(1.1 > Math.exp(-llSum / ll.blocksCount()));
  }

}
