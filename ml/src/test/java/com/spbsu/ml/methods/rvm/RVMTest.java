package com.spbsu.ml.methods.rvm;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.random.FastRandom;
import gnu.trove.iterator.TIntIterator;
import gnu.trove.set.TIntSet;
import gnu.trove.set.hash.TIntHashSet;
import junit.framework.TestCase;

/**
 * Created by noxoomo on 04/06/15.
 */
public class RVMTest extends TestCase {
  public void testActiveIndices() {
    TIntSet active = new TIntHashSet();
    RVMCache.ActiveIndicesSet  activeIndices = new RVMCache.ActiveIndicesSet(100,new FastRandom(10));

    activeIndices.addToActive(0);
    active.add(0);
    assertEquals(activeIndices.size(), 1);
    activeIndices.addToActive(4);
    active.add(4);
    assertEquals(activeIndices.size(), 2);

    TIntHashSet in = new TIntHashSet(activeIndices.activeIndices());
    assertEquals(in,active);

    active.add(9);
    active.add(2);
    active.add(22);
    active.add(3);
    active.add(11);
    activeIndices.addToActive(9);
    activeIndices.addToActive(2);
    activeIndices.addToActive(22);
    activeIndices.addToActive(3);
    activeIndices.addToActive(11);

    in = new TIntHashSet(activeIndices.activeIndices());
    assertEquals(in,active);

    active.remove(4);
    active.add(33);
    active.remove(22);


    activeIndices.removeFromActive(4);
    activeIndices.addToActive(33);
    activeIndices.removeFromActive(22);

    in = new TIntHashSet(activeIndices.activeIndices());
    assertEquals(in,active);

    {
      int[] byIter = new int[activeIndices.size()];
      TIntIterator it = activeIndices.activeIterator();
      int i=0;
      while (it.hasNext()) {
        byIter[i++] = it.next();
      }

      in = new TIntHashSet(byIter);
      assertEquals(in,active);
    }
  }

  public void testDotProducts() {
    FastRandom rand = new FastRandom();
    Mx learn = new VecBasedMx(1023, 117);
    Vec target = new ArrayVec(1023);
    for (int i=0; i < target.dim();++i)
      target.set(i,rand.nextDouble());

    Vec ones = new ArrayVec(1023);
    VecTools.fill(ones,1.0);

    RVMCache.DotProductsCache featuresProducts = new RVMCache.DotProductsCache(learn,target);

    assertEquals(featuresProducts.featuresProduct(10,20), VecTools.multiply(learn.col(10),learn.col(20)));
    assertEquals(featuresProducts.featuresProduct(10,20), VecTools.multiply(learn.col(10),learn.col(20)));
    assertEquals(featuresProducts.featuresProduct(20,10), VecTools.multiply(learn.col(20),learn.col(10)));


    assertEquals(featuresProducts.featuresProduct(10,117), VecTools.multiply(learn.col(10),ones));
    assertEquals(featuresProducts.featuresProduct(117,10), VecTools.multiply(learn.col(10),ones));

    assertEquals(featuresProducts.targetProducts(20), VecTools.multiply(learn.col(20),target));
    assertEquals(featuresProducts.targetProducts(20), VecTools.multiply(learn.col(20),target));
    assertTrue(Math.abs(featuresProducts.targetProducts(117)-VecTools.multiply(ones,target)) < 1e-9);
    assertTrue(Math.abs(featuresProducts.targetProducts(117) - VecTools.multiply(ones, target)) < 1e-9);

    for (int i=0; i <= learn.columns();++i) {
      for (int j=0;j <= learn.columns();++j)
        assertEquals(featuresProducts.featuresProduct(i,j),featuresProducts.featuresProduct(j,i));
    }
  }

}