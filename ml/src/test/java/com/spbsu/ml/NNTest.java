package com.spbsu.ml;

import com.spbsu.commons.func.Action;
import com.spbsu.commons.func.Computable;
import com.spbsu.commons.func.Processor;
import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.math.io.Vec2CharSequenceConverter;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.SingleValueVec;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.seq.*;
import com.spbsu.ml.data.tools.Pool;
import com.spbsu.ml.data.tools.PoolByRowsBuilder;
import com.spbsu.ml.func.generic.Log;
import com.spbsu.ml.func.generic.Sum;
import com.spbsu.ml.loss.CompositeFunc;
import com.spbsu.ml.loss.DSSumFuncComposite;
import com.spbsu.ml.loss.LL;
import com.spbsu.ml.loss.blockwise.BlockwiseMLL;
import com.spbsu.ml.meta.DataSetMeta;
import com.spbsu.ml.meta.FeatureMeta;
import com.spbsu.ml.meta.items.FakeItem;
import com.spbsu.ml.meta.items.QURLItem;
import com.spbsu.ml.methods.StochasticGradientDescent;
import com.spbsu.ml.models.NeuralSpider;
import com.spbsu.ml.models.nn.LayeredNetwork;
import com.spbsu.ml.models.nn.NFANetwork;
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
    final DSSumFuncComposite<FakeItem> target = new DSSumFuncComposite<>(pool.data(), ll, new Computable<FakeItem, FuncC1>() {
      @Override
      public FuncC1 compute(final FakeItem argument) {
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
      final NFANetwork<Character> nfa = new NFANetwork<>(rng, 0., 5, new CharSeqAdapter("ab"));
      final FuncC1 aba = nfa.decisionByInput(new CharSeqAdapter("aba"));
      Assert.assertEquals(0.2 + 0.16 + 0.128, aba.value(new ArrayVec(nfa.dim())), 0.0001);
  }

  @Test
  public void testSeqGradient1() {
    String message = "";

    final NFANetwork<Character> nfa = new NFANetwork<>(rng, 0., 2, new CharSeqAdapter("ab"));

    final NeuralSpider.NeuralNet ab = nfa.decisionByInput(new CharSeqAdapter("ab"));
    final NeuralSpider.NeuralNet ba = nfa.decisionByInput(new CharSeqAdapter("ba"));
    final Vec x = new ArrayVec(1, 2);
    message += nfa.ppState(ab.state(x), new CharSeqAdapter("ab"));
    final Vec gradientAb = ab.gradient(x);
    message += nfa.ppState(ba.state(x), new CharSeqAdapter("ba"));
    final Vec gradientBa = ba.gradient(x);
    // composite result:  1/(1+e^x)*(1 + e^x/(1+e^y))
    message += "\nor: " + x + "\n"
            + "ab: " + gradientAb + "\n"
            + "ba: " + gradientBa + "\n";
    // composite gradient by x: -e^x/(1+e^x)*1/(1+e^x)*e^y/(1+e^y)
    Assert.assertEquals(message, -0.17318, gradientAb.get(0), 0.00001);
    // composite gradient by x: -e^y/(1+e^y)*1/(1+e^y)*1/(1+e^x)
    Assert.assertEquals(message, -0.076756, gradientAb.get(1), 0.00001);
    Assert.assertTrue(message, VecTools.equals(gradientAb, gradientBa));
  }

  @Test
  public void testSeqGradient2() {
    String message = "";

    final NFANetwork<Character> nfa = new NFANetwork<>(rng, 0., 3, new CharSeqAdapter("ab"));

    final NeuralSpider.NeuralNet ab = nfa.decisionByInput(new CharSeqAdapter("ab"));
    final NeuralSpider.NeuralNet ba = nfa.decisionByInput(new CharSeqAdapter("ba"));
    final Vec x = new ArrayVec(1,0, 0,1,  0,1, 0,0);
    message += nfa.ppState(ab.state(x), new CharSeqAdapter("ab"));
    final Vec gradientAb = ab.gradient(x);
    message += nfa.ppState(ba.state(x), new CharSeqAdapter("ba"));
    final Vec gradientBa = ba.gradient(x);
    message += "\nor: " + x + "\n"
            + "ab: " + gradientAb + "\n"
            + "ba: " + gradientBa + "\n";
    Assert.assertTrue(message, VecTools.equals(gradientAb, new Vec2CharSequenceConverter().convertFrom("8 -0.11105 -0.01512 0 0 -0.02588 -0.07035 -0.02355 -0.02355"), 0.00001));
    Assert.assertTrue(message, VecTools.equals(gradientBa, new Vec2CharSequenceConverter().convertFrom("8 -0.02588 -0.00952 -0.02588 -0.07035 -0.0354 -0.09622 0 0"), 0.00001));
  }

  @Test
  public void testSeqGradient3() {
    String message = "";

    final NFANetwork<Character> nfa = new NFANetwork<>(rng, 0., 4, new CharSeqAdapter("ab"));

    final NeuralSpider.NeuralNet ab = nfa.decisionByInput(new CharSeqAdapter("ab"));
    final NeuralSpider.NeuralNet ba = nfa.decisionByInput(new CharSeqAdapter("ba"));
    final Vec x = new ArrayVec(
            1,0,0,
            0,1,0,
            0,0,1,

            0,1,0,
            0,0,1,
            0,0,0);
    message += nfa.ppState(ab.state(x), new CharSeqAdapter("ab"));
    final Vec gradientAb = ab.gradient(x);
    message += nfa.ppState(ba.state(x), new CharSeqAdapter("ba"));
    final Vec gradientBa = ba.gradient(x);
    message += "\nor: " + x + "\n"
            + "ab: " + gradientAb + "\n"
            + "ba: " + gradientBa + "\n";
    Assert.assertTrue(message, VecTools.equals(gradientAb, new Vec2CharSequenceConverter().convertFrom("18 -0.11105 -0.01512 -0.08577 0 0 0 0 0 0 -0.02588 -0.07035 -0.02588 -0.02355 -0.02355 -0.06401 0 0 0"), 0.00001));
  }

  @Test
  public void testSimpleSeq() {
    final int statesCount = 3;
    final CharSeqAdapter alpha = new CharSeqAdapter("ab");
    final NFANetwork<Character> nfa = new NFANetwork<>(rng, 0.1, statesCount, alpha);

    final PoolByRowsBuilder<FakeItem> pbuilder = new PoolByRowsBuilder<>(DataSetMeta.ItemType.FAKE);
    pbuilder.allocateFakeFeatures(1, FeatureMeta.ValueType.CHAR_SEQ);
    pbuilder.allocateFakeTarget(FeatureMeta.ValueType.INTS);
    pbuilder.setFeature(0, new CharSeqAdapter("abba"));
    pbuilder.setTarget(0, 1);
    pbuilder.nextItem();
    pbuilder.setFeature(0, new CharSeqAdapter("baba"));
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
            final NeuralSpider.NeuralNet abba = nfa.decisionByInput(new CharSeqAdapter("abba"));
            System.out.println(nfa.ppState(abba.state(vec), new CharSeqAdapter("abba")));
          }
          {
            System.out.println("Negative: ");
            final NeuralSpider.NeuralNet baba = nfa.decisionByInput(new CharSeqAdapter("baba"));
            System.out.println(nfa.ppState(baba.state(vec), new CharSeqAdapter("baba")));
          }
        }
      }
    };
    gradientDescent.addListener(pp);
    final LL logit = pool.target(LL.class);
    final DSSumFuncComposite<FakeItem> target = new DSSumFuncComposite<>(pool.data(), logit, new Computable<FakeItem, FuncC1>() {
      @Override
      public FuncC1 compute(final FakeItem argument) {
        final CharSeq seq = pool.feature(0, argument.id);
        return nfa.decisionByInput(seq);
      }
    });
    final DSSumFuncComposite<FakeItem>.Decision fit = gradientDescent.fit(pool.data(), target);

    final Vec solution = fit.x;
    nfa.ppSolution(solution);
    {
      System.out.println("Positive: ");
      final NeuralSpider.NeuralNet abba = nfa.decisionByInput(new CharSeqAdapter("abba"));
      System.out.println(nfa.ppState(abba.state(solution), new CharSeqAdapter("abba")));
      Assert.assertTrue(abba.value(solution) > 0.95);
    }
    {
      System.out.println("Negative: ");
      final NeuralSpider.NeuralNet baba = nfa.decisionByInput(new CharSeqAdapter("baba"));
      System.out.println(nfa.ppState(baba.state(solution), new CharSeqAdapter("baba")));
      Assert.assertTrue(baba.value(solution) < 0.05);
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
    final LL ll = pool.target(LL.class);
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
    final DSSumFuncComposite<FakeItem> target = new DSSumFuncComposite<>(pool.data(), ll, new Computable<FakeItem, FuncC1>() {
      @Override
      public FuncC1 compute(final FakeItem argument) {
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
        final int paramsDim = (statesCount - 1) * (statesCount - 1);
        for (int c = 0; c < alpha.length(); c++) {
          final VecBasedMx mx = new VecBasedMx(statesCount - 1, cursor.sub(c * paramsDim, paramsDim));
          VecTools.fillUniform(mx, rng, 5 / (statesCount - 1));
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
    final LL ll = pool.target(LL.class);
    final ArrayVec initial = new ArrayVec(network.dim());
    gradientDescent.init(initial);
//    digIntoSolution(pool, network, ll, initial, "www.yandex.ru/yandsearch?text=xyu.htm", "www.yandex.ru");

    final DSSumFuncComposite<FakeItem> target = new DSSumFuncComposite<>(pool.data(), ll, new Computable<FakeItem, FuncC1>() {
      @Override
      public FuncC1 compute(final FakeItem argument) {
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
                if (nextClass > 0 != this.next)
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
    final NFANetwork<Character> network = new NFANetwork<>(rng, 0.1, statesCount, alpha);
    final StochasticGradientDescent<FakeItem> gradientDescent = new StochasticGradientDescent<FakeItem>(rng, 4, 1000000, 1) {
      @Override
      public void init(Vec cursor) {
        final int paramsDim = (statesCount - 1) * (statesCount - 1);
        for (int c = 0; c < alpha.length(); c++) {
          final VecBasedMx mx = new VecBasedMx(statesCount - 1, cursor.sub(c * paramsDim, paramsDim));
          VecTools.fillUniform(mx, rng);
          for (int i = 0; i < mx.rows(); i++) {
            mx.set(i, i, 5);
          }
        }
      }

      @Override
      public void normalizeGradient(Vec grad) {
//        for (int i = 0; i < grad.length(); i++) {
//          if (Math.abs(grad.get(i)) < 0.001)
//            grad.set(i, 0);
//        }
      }
    };
    final LL ll = pool.target(LL.class);
    final Action<Vec> pp = new Action<Vec>() {
      int index = 0;
      @Override
      public void invoke(Vec vec) {
        if (++index % 10000 == 1) {
          double sum = 0;
          int count = 0;
          int negative = 0;
          for (int i = 0; i < 1000; i++, count++) {
            final double value = ll.block(i).value(new SingleValueVec(network.decisionByInput((CharSeq) pool.feature(0, i)).value(vec)));
            sum += value;
            if (Math.exp(-value) > 2)
              negative++;
          }
          System.out.println(index + " ll: " + Math.exp(-sum / count) + " prec: " + (count - negative)/(double)count);
//          network.ppSolution(vec, 'h');
//          network.ppSolution(vec, 't');
//          network.ppSolution(vec, 'm');
//          network.ppSolution(vec, 'l');
//          network.ppSolution(vec, '.');
        }
      }
    };
    gradientDescent.addListener(pp);
    final DSSumFuncComposite<FakeItem> target = new DSSumFuncComposite<>(pool.data(), ll, new Computable<FakeItem, FuncC1>() {
      @Override
      public FuncC1 compute(final FakeItem argument) {
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
//
//    final List<CharSequence> data = new ArrayList<>();
//    final TIntArrayList classes = new TIntArrayList();
//
//    final int T = 1000000;
//    double perplexity = 0;
//    int perCount = 0;
//    final NFANetwork<Character> nfaNetwork = new NFANetwork<>(rng, 0.2, 6, alpha);
//    final TDoubleArrayList lls = new TDoubleArrayList();
//    final int coupleSize = 100;
//    Vec totalGrad = new ArrayVec(nfaNetwork.dim());
//    Executor executor = ThreadTools.createBGExecutor(NNTest.class.getName(), coupleSize);
//    for (int t = 0; t < T; t++) {
//      final Vec gradient = new ArrayVec(nfaNetwork.dim());
//      final CountDownLatch latch = new CountDownLatch(coupleSize);
//      for (int j = 0; j < coupleSize; j++) {
//        executor.execute(new Runnable() {
//          @Override
//          public void run() {
//            int index;
//            boolean good = rng.nextBoolean();
//            do {
//              index = rng.nextInt(data.size());
//            } while (classes.get(index) == 1 != good);
//            final CharSeq seq = CharSeqAdapter.create(data.get(index));
//            final FuncC1 target = good ? new LogSigmoid(-1, 0) : new LogSigmoid(1, 0);
//            final Vec myGradient = nfaNetwork.parametersGradient(seq, target);
//            synchronized (gradient) {
//              VecTools.append(gradient, myGradient);
//            }
//            latch.countDown();
//          }
//        });
//      }
//      latch.await();
//
//      VecTools.scale(gradient, 1. / coupleSize);
//      VecTools.scale(gradient, 0.1 * 100. / sqrt(10000. + t));
////      { // l1 regularization
////        for (int j = 0; j < gradient.length(); j++) {
////          if (Math.abs(gradient.at(j)) < 0.0001)
////            gradient.set(j, 0);
////        }
////      }
//      final double cos = VecTools.multiply(totalGrad, gradient) / (MathTools.EPSILON + VecTools.norm(totalGrad)) / VecTools.norm(gradient);
//      if (Double.isNaN(cos))
//        continue;
//      VecTools.scale(totalGrad, 0.99);
//      VecTools.append(totalGrad, gradient);
//      VecTools.scale(gradient, (1. + cos));
//
////      VecTools.scale(gradient, 0.2);
//      VecTools.append(nfaNetwork.allWeights(), gradient);
//
//      final int index = rng.nextInt(data.size());
//      final boolean good = classes.get(index) == 1;
//      final double value = nfaNetwork.compute(CharSeqAdapter.create(data.get(index))).get(0);
//      {
//        final FuncC1 target = good ? new LogSigmoid(-1, 0) : new LogSigmoid(1, 0);
//        final double ll = target.value(new SingleValueVec(value));
//        lls.add(ll);
//        perCount++;
//        perplexity += ll;
//        if (perCount > 1000) {
//          perplexity -= lls.get(lls.size() - 1001);
//          perCount--;
//        }
//      }
////      System.out.println(gradient);
//      if ((t + 1) % 1000 == 0 || T - t < 10) {
//        double per = perplexity / perCount;
//        per = exp(-per);
//        System.out.print("Iteration: " + (t + 1) + " v: " + value + " good: " + good + " perplexity: " + per);
//        System.out.println(" cos(prev,grad): " + cos + " grad norm: " + VecTools.norm(gradient));
//      }
//    }
//    perplexity /= perCount;
//    perplexity = exp(-perplexity);
//    System.out.println(perplexity);
//    Assert.assertTrue(perplexity < 1.01);
  }

  @Test
  public void testFTRLSeqConvergence() throws Exception {
//    final List<CharSequence> data = new ArrayList<>();
//    final TIntArrayList classes = new TIntArrayList();
//
//    CharSeqTools.processLines(
//            new InputStreamReader(new GZIPInputStream(new FileInputStream("/Users/solar/tree/java/relpred/trunk/relpred/main/tests/data/in/train.txt.gz"))),
//            new Processor<CharSequence>() {
//              CharSequence[] parts = new CharSequence[2];
//
//              @Override
//              public void process(CharSequence arg) {
//                CharSeqTools.split(arg, '\t', parts);
//                data.add(parts[0]);
//                classes.add(Integer.parseInt(CharSeqTools.split(parts[1], ':')[1].toString()));
//              }
//            });
//    final int T = 1000000;
//
//    double perplexity = 0;
//    int perCount = 0;
//    final NFANetwork<Character> nfaNetwork = new NFANetwork<>(rng, 0.2, 6, new CharSeqArray('U', 'L', 'H', 'C', 'S', 'N', 'R', 'F', 'V', 'O'));
//    final TDoubleArrayList lls = new TDoubleArrayList();
//    final int coupleSize = 10;
//    final Vec z = new ArrayVec(nfaNetwork.dim());
//    final Vec n = new ArrayVec(nfaNetwork.dim());
//    double LAMBDA1 = 0.001/nfaNetwork.dim();
//    double LAMBDA2 = 0.01/nfaNetwork.dim();
//    double ALPHA = 1;
//    double BETA = 1;
//
//    final Executor executor = ThreadTools.createBGExecutor(NNTest.class.getName(), coupleSize);
//    final Vec totalGrad = new ArrayVec(nfaNetwork.dim());
//    final Vec weightedSteps = new ArrayVec(nfaNetwork.dim());
//
//    for (int t = 0; t < T; t++) {
//      final Vec gradient = new ArrayVec(nfaNetwork.dim());
//
//      final int index = rng.nextInt(data.size());
//      final boolean good = classes.get(index) == 1;
//      final double p = 1/(1 + exp(nfaNetwork.compute(CharSeqAdapter.create(data.get(index))).get(0)));
//
//      for (int i = 0; i < nfaNetwork.allWeights().length(); i++) {
//        if (abs(z.get(i)) >= LAMBDA1) {
//          nfaNetwork.allWeights().set(i,
//                  -(z.get(i) - signum(z.get(i)) * LAMBDA1) / ((BETA + sqrt(n.get(i)))/ALPHA + LAMBDA2)
//          );
//        }
//        else nfaNetwork.allWeights().set(i, 0);
//      }
//
//      final CountDownLatch latch = new CountDownLatch(coupleSize);
//      for (int j = 0; j < coupleSize; j++) {
//        executor.execute(new Runnable() {
//          @Override
//          public void run() {
//            int index;
//            boolean good = rng.nextBoolean();
//            do {
//              index = rng.nextInt(data.size());
//            } while (classes.get(index) == 1 != good);
//            final CharSeq seq = CharSeqAdapter.create(data.get(index));
//            final FuncC1 target = good ? new LogSigmoid(-1, 0) : new LogSigmoid(1, 0);
//            final Vec myGradient = nfaNetwork.parametersGradient(seq, target);
//            synchronized (gradient) {
//              VecTools.append(gradient, myGradient);
//            }
//            latch.countDown();
//          }
//        });
//      }
//      latch.await();
//
//      VecTools.scale(gradient, 1. / coupleSize);
//      for (int i = 0; i < nfaNetwork.dim(); i++) {
//        final double g = gradient.get(i);
//        final double sigma = 1/ALPHA * (sqrt(n.get(i) + g * g) - sqrt(n.get(i)));
//        z.adjust(i, g - sigma * nfaNetwork.allWeights().get(i));
//        n.adjust(i, g * g);
//      }
//
////      VecTools.append(nfaNetwork.allWeights(), gradient);
//
//      {
//        final FuncC1 target = good ? new LogSigmoid(-1, 0) : new LogSigmoid(1, 0);
//        final double ll = target.value(new SingleValueVec(p));
//        lls.add(ll);
//        perCount++;
//        perplexity += ll;
//        if (perCount > 1000) {
//          perplexity -= lls.get(lls.size() - 1001);
//          perCount--;
//        }
//      }
////      System.out.println(gradient);
//      if ((t + 1) % 1000 == 0 || T - t < 10) {
//        double per = perplexity / perCount;
//        per = exp(-per);
//        System.out.print("Iteration: " + (t + 1) + " v: " + p + " good: " + good + " perplexity: " + per);
////        System.out.print(" cos(prev,grad): " + cos + " grad norm: " + VecTools.norm(gradient));
//        System.out.println();
//      }
//    }
//    perplexity /= perCount;
//    perplexity = exp(-perplexity);
//    System.out.println(perplexity);
//    Assert.assertTrue(perplexity < 1.01);
  }

  private void digIntoSolution(Pool<FakeItem> pool, NFANetwork<Character> network, LL ll, Vec solution, String positiveExample, String negativeExample) {
    if (positiveExample != null) {
      System.out.println("Positive: ");
      final CharSeqAdapter input = new CharSeqAdapter(positiveExample);
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
      final CharSeqAdapter input = new CharSeqAdapter(negativeExample);
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
    final Vec vals = new ArrayVec(pool.size());
    int count = 0, negative = 0;
    for (int i = 0; i < ll.blocksCount(); i++) {
      final CharSeq input = pool.feature(0, i);
      final double pX = network.decisionByInput(input).value(solution);
      vals.set(i, pX);
      final double value = Math.exp(-ll.block(i).value(new SingleValueVec(pX)));
      count++;
      if (2 < value) {
        negative++;
        System.out.println("Input: [" + input + "]");
        final NeuralSpider.NeuralNet net = network.decisionByInput(input);
        System.out.println(network.ppState(net.state(solution), input));
        System.out.println();
      }
    }
    System.out.println(Math.exp(-ll.value(vals) / ll.dim()) + " " + (count - negative) / (double)count);
    Assert.assertTrue(1.1 > Math.exp(-ll.value(vals) / ll.dim()));
  }


  @Test
  public void testCompositeFunc() {
    FuncC1 g = new Sum() {
      @Override
      public int dim() {
        return 2;
      }
    };
    Trans a = new Trans.Stub() {
      @Override
      public int xdim() {
        return 3;
      }

      @Override
      public int ydim() {
        return 2;
      }

      @Override
      public Trans gradient() {
        return new Trans.Stub() {
          @Override
          public int xdim() {
            return 3;
          }

          @Override
          public int ydim() {
            return 6;
          }

          @Override
          public Vec trans(Vec arg) {
            Mx mx = new VecBasedMx(2, 3);
            mx.set(0,0, arg.get(1));
            mx.set(0,1, arg.get(0));
            mx.set(0,2, 0);
            mx.set(1,0, 0);
            mx.set(1,1, 1);
            mx.set(1,2, 1);
            return mx;
          }
        };
      }

      @Override
      public Vec trans(Vec argument) {
        Vec to = new ArrayVec(ydim());
        to.adjust(0, argument.get(0) * argument.get(1));
        to.adjust(1, argument.get(1) + argument.get(2));
        return to;
      }
    };
    Trans b = new Trans.Stub() {
      @Override
      public Vec trans(Vec argument) {
        Vec to = new ArrayVec(ydim());
        to.adjust(0, argument.get(0) + argument.get(1));
        to.adjust(1, argument.get(0) + 2);
        to.adjust(2, argument.get(1) + 4);
        return to;
      }

      @Override
      public int xdim() {
        return 2;
      }

      @Override
      public int ydim() {
        return 3;
      }

      @Override
      public Trans gradient() {
        return new Trans.Stub() {
          @Override
          public int xdim() {
            return 2;
          }

          @Override
          public int ydim() {
            return 6;
          }

          @Override
          public Vec trans(Vec arg) {
            Mx mx = new VecBasedMx(3, 2);
            mx.set(0,0, 1);
            mx.set(0,1, 1);
            mx.set(1,0, 1);
            mx.set(1,1, 0);
            mx.set(2,0, 0);
            mx.set(2,1, 1);
            return mx;
          }
        };
      }
    };
    CompositeFunc compositeFunc = new CompositeFunc(g, a, b);
    Assert.assertEquals(31, compositeFunc.value(new ArrayVec(2, 3)), 0);
    Assert.assertEquals(new ArrayVec(10, 5), compositeFunc.gradient(new ArrayVec(2, 3)));
  }

  //  @Test
  public void testBlockwiseMLLConvergence() {
    int classesCount = 3;
    int inputDim = 3;

    final PoolByRowsBuilder<FakeItem> pbuilder = new PoolByRowsBuilder<>(DataSetMeta.ItemType.FAKE);
    pbuilder.allocateFakeFeatures(inputDim, FeatureMeta.ValueType.VEC);
    pbuilder.allocateFakeTarget(FeatureMeta.ValueType.INTS);
    for (int i = 0; i < 10000; i++) {
      final Vec next = new ArrayVec(inputDim);
      for (int j = 0; j < next.dim(); j++)
        next.set(j, rng.nextInt(classesCount));
      pbuilder.setFeatures(0, next);
      pbuilder.setTarget(0, (int) next.get(0));
      pbuilder.nextItem();
    }

    final Pool<FakeItem> pool = pbuilder.create();
    final LayeredNetwork network = new LayeredNetwork(rng, 0., inputDim, 3, 3, classesCount - 1);
    final StochasticGradientDescent<FakeItem> gradientDescent = new StochasticGradientDescent<FakeItem>(rng, 4, 1000000, 0.8) {
      public void init(Vec cursor) {
        VecTools.fillUniform(cursor, rng);
      }
    };
    final Mx data = pool.vecData().data();
    final BlockwiseMLL blockwiseMLL = pool.target(BlockwiseMLL.class);
    final DSSumFuncComposite<FakeItem> target = new DSSumFuncComposite<>(pool.data(), blockwiseMLL, new Computable<FakeItem, FuncC1>() {
      @Override
      public FuncC1 compute(final FakeItem argument) {
        final Vec row = data.row(argument.id);
        return network.decisionByInput(row);
      }
    });
    final DSSumFuncComposite<FakeItem>.Decision decision = gradientDescent.fit(pool.data(), target);
    System.out.println(decision.x);
    final Mx vals = new VecBasedMx(pool.size(), classesCount - 1);
    for (int i = 0; i < vals.rows(); i++) {
      for (int j = 0; j < vals.columns(); j++) {
        vals.set(i, j, decision.compute(pool.data().at(i)).get(j));
      }
    }
    double resultValue = blockwiseMLL.transformResultValue(-blockwiseMLL.value(vals));
    System.out.println(resultValue);
    Assert.assertTrue(1.1 > resultValue);
  }


  //@Test
  public void testSimpleBlockwiseMLL() {
    final PoolByRowsBuilder<FakeItem> pbuilder = new PoolByRowsBuilder<>(DataSetMeta.ItemType.FAKE);
    pbuilder.allocateFakeFeatures(3, FeatureMeta.ValueType.VEC);
    pbuilder.allocateFakeTarget(FeatureMeta.ValueType.INTS);
    final Vec[] vecs = new Vec[3];
    vecs[0] = new ArrayVec(2, 10, 10);
    vecs[1] = new ArrayVec(11, 1, 2);
    vecs[2] = new ArrayVec(1, 1, 1);
    for (int i = 0; i < vecs.length; i++) {
      pbuilder.setFeatures(0, vecs[i]);
      pbuilder.setTarget(0, i);
      pbuilder.nextItem();
    }

    final Pool<FakeItem> pool = pbuilder.create();

    final LayeredNetwork network = new LayeredNetwork(rng, 0., 3, 3, 3, 2);
    final StochasticGradientDescent<FakeItem> gradientDescent = new StochasticGradientDescent<FakeItem>(rng, 4, 1000, 0.8) {
      public void init(Vec cursor) {
        VecTools.fillUniform(cursor, rng);
      }
    };

    final Action<Vec> pp = new Action<Vec>() {
      int index = 0;

      @Override
      public void invoke(Vec vec) {
        if (++index == 1) {
          for (int i = 0; i < vecs.length; i++) {
            System.out.println("Class: " + i);
            final NeuralSpider.NeuralNet neuralNet = network.decisionByInput(vecs[i]);
            System.out.println(neuralNet.state(vec));
          }
        }
      }
    };
    gradientDescent.addListener(pp);

    final Mx data = pool.vecData().data();
    final BlockwiseMLL blockwiseMLL = pool.target(BlockwiseMLL.class);
    final DSSumFuncComposite<FakeItem> target = new DSSumFuncComposite<>(pool.data(), blockwiseMLL, new Computable<FakeItem, Trans>() {
      @Override
      public Trans compute(final FakeItem argument) {
        final Vec row = data.row(argument.id);
        return network.decisionByInput(row);
      }
    });
    final DSSumFuncComposite<FakeItem>.Decision decision = gradientDescent.fit(pool.data(), target);
    System.out.println(decision.x);

    for (int i = 0; i < vecs.length; i++) {
      System.out.println("Class: " + i);
      System.out.println(decision.compute(pool.data().at(i)));
    }
  }
}
