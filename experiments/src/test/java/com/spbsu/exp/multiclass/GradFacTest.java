package com.spbsu.exp.multiclass;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.MxTools;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.util.Pair;
import com.spbsu.ml.data.tools.DataTools;
import com.spbsu.ml.data.tools.Pool;
import com.spbsu.ml.data.tools.SubPool;
import com.spbsu.ml.factorization.OuterFactorization;
import com.spbsu.ml.factorization.impl.ALS;
import com.spbsu.ml.factorization.impl.ElasticNetFactorization;
import com.spbsu.ml.factorization.impl.SVDAdapterEjml;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.blockwise.BlockwiseMLLLogit;
import com.spbsu.ml.meta.FeatureMeta;
import com.spbsu.ml.meta.impl.FakeTargetMeta;
import com.spbsu.ml.testUtils.TestResourceLoader;
import junit.framework.TestCase;

import java.io.IOException;

/**
 * User: qdeee
 * Date: 27.02.15
 */
public class GradFacTest extends TestCase {
  private static Pool<?> learn;
  private static Pool<?> test;
  private static Mx S;

  private synchronized static void init() throws IOException {
    if (learn == null || test == null) {
      final Pool<?> pool = TestResourceLoader.loadPool("multiclass/ds_letter/letter.tsv.gz");
      pool.addTarget(new FakeTargetMeta(pool.vecData(), FeatureMeta.ValueType.INTS),
          VecTools.toIntSeq(pool.target(L2.class).target)
      );
      final int[][] idxs = DataTools.splitAtRandom(pool.size(), new FastRandom(100500), 0.8, 0.5);
      learn = new SubPool<>(pool, idxs[0]);
      test = new SubPool<>(pool, idxs[1]);
    }
  }

  @Override
  protected void setUp() throws Exception {
    init();
  }


  public void testGradMxApproxSVDN() throws Exception {
    final BlockwiseMLLLogit globalLoss = learn.target(BlockwiseMLLLogit.class);
    final Mx gradient = (Mx) globalLoss.gradient(new ArrayVec(globalLoss.dim()));
    double time = System.currentTimeMillis();

    for (int factorDim = gradient.columns(); factorDim >= 1; factorDim--)
    {
      final Pair<Vec, Vec> pair = new SVDAdapterEjml(factorDim).factorize(gradient);
      final Mx h = (Mx) pair.getFirst();
      final Mx b = (Mx) pair.getSecond();
      System.out.println("factor dim: " + factorDim);
      System.out.println("time: " + ((System.currentTimeMillis() - time) / 1000));
      final Mx afterFactor = MxTools.multiply(h, MxTools.transpose(b));
      System.out.println("||h|| = " + VecTools.norm(h) + ", ||b|| = " + VecTools.norm(b) + ", l2 = " + VecTools.distance(gradient, afterFactor) + ", l1 = " + VecTools.distanceL1(gradient, afterFactor));
      System.out.println();
    }
  }
  public void testElasticNetGradFac() throws Exception {
    final BlockwiseMLLLogit globalLoss = learn.target(BlockwiseMLLLogit.class);
    final Mx gradient = (Mx) globalLoss.gradient(new ArrayVec(globalLoss.dim()));
    final ElasticNetFactorization elasticNetFactorization = new ElasticNetFactorization(1, 1e-2, 0.0, 0.0);
    final Pair<Vec, Vec> pair = elasticNetFactorization.factorize(gradient);
    final Vec h = pair.getFirst();
    final Vec b = pair.getSecond();
    final Mx afterFactor = VecTools.outer(h, b);
    System.out.println("||h|| = " + VecTools.norm(h) + ", ||b|| = " + VecTools.norm(b) + ", l2 = " + VecTools.distance(gradient, afterFactor) + ", l1 = " + VecTools.distanceL1(gradient, afterFactor));
    System.out.println();
  }

  public void testSimpleMx() throws Exception {
    final Mx mx = genUniformRandMx(5, 3, 100500);

    final ElasticNetFactorization elasticNetFactorization = new ElasticNetFactorization(1, 1e-1, 0.0, 0.0);
    final Pair<Vec, Vec> pair = elasticNetFactorization.factorize(mx);
    final Vec h = pair.getFirst();
    final Vec b = pair.getSecond();
    final Mx afterFactor = VecTools.outer(h, b);
    System.out.println("||h|| = " + VecTools.norm(h) + ", ||b|| = " + VecTools.norm(b) + ", l2 = " + VecTools.distance(mx, afterFactor) + ", l1 = " + VecTools.distanceL1(mx, afterFactor));
  }

  private static Mx genUniformRandMx(final int m, final int n, final int seed) {
    final Mx mx = new VecBasedMx(m, n);
    final FastRandom fastRandom = new FastRandom(seed);
    for (int i = 0; i < mx.dim(); i++) {
      mx.set(i, fastRandom.nextDouble());
    }
    return mx;
  }

  public void testDifferentMethods() throws Exception {
    final Mx mx = genUniformRandMx(50, 30, 100500);

    applyFactorMethod(mx, new ALS(15));
    applyFactorMethod(mx, new SVDAdapterEjml());
    final double lambda = 0.0000015;
    applyFactorMethod(mx, new ElasticNetFactorization(20, 1e-10, 0, 0));
  }

  private static void applyFactorMethod(final Mx x, final OuterFactorization method) {
    final Pair<Vec, Vec> pair = method.factorize(x);
    final Vec h = pair.getFirst();
    final Vec b = pair.getSecond();
    final double normB = VecTools.norm(b);
    VecTools.scale(b, 1 / normB);
    VecTools.scale(h, normB);
    final Mx afterFactor = VecTools.outer(h, b);
    System.out.println(method.getClass().getSimpleName() + ": ||h|| = " + VecTools.norm(h) + ", ||b|| = " + VecTools.norm(b) + ", l2 = " + VecTools.distance(x, afterFactor) + ", l1 = " + VecTools.distanceL1(x, afterFactor));
  }
}
