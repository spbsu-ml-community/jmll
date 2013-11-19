package com.spbsu.ml.trees;

import junit.framework.TestCase;

//import com.spbsu.ml.methods.trees.MultiLLClassificationLeaf;

/**
 * User: solar
 * Date: 17.09.13
 * Time: 9:45
 */
public class MultiClassTest extends TestCase {
//  public void testAlpha() {
//    final Vec target = new ArrayVec(
//            0, 0, 1, 1, 1
//    );
//    final Mx data = new VecBasedMx(1, new ArrayVec(
//            1, 1, 1, 0, 0
//    ));
//    final DataSetImpl ds = new DataSetImpl(data, target);
//    final BFGrid grid = GridTools.medianGrid(ds, 2);
//    assertEquals(grid.size(), 1);
//    final BinarizedDataSet bds = new BinarizedDataSet(ds, grid);
//    final MultiLLClassificationLeaf leaf = new MultiLLClassificationLeaf(bds, new Vec[]{new ArrayVec(target.dim()), new ArrayVec(target.dim())}, target, VecTools.fill(new ArrayVec(target.dim()), 1));
//    assertTrue(abs(abs(leaf.alpha()) - abs(log(3. / 2.))) < MathTools.EPSILON);
////    assertTrue(abs(leaf.score() - (-2*3*log(1.+2/3.)-2*2*log(1.+3/2.) - 2 * target.dim() * log(0.5))) < 0.01);
//  }
//
//  public void testScores() {
//    final Vec target = new ArrayVec(
//            0, 0, 1, 1, 1
//    );
//    final Mx data = new VecBasedMx(1, new ArrayVec(
//            1, 1, 1, 0, 0
//    ));
//    final DataSetImpl ds = new DataSetImpl(data, target);
//    final BFGrid grid = GridTools.medianGrid(ds, 2);
//    assertEquals(grid.size(), 1);
//    final BinarizedDataSet bds = new BinarizedDataSet(ds, grid);
//    final MultiLLClassificationLeaf leaf = new MultiLLClassificationLeaf(bds, new Vec[]{new ArrayVec(target.dim()), new ArrayVec(target.dim())}, target, VecTools.fill(new ArrayVec(target.dim()), 1));
//    final double[] scores = new double[1];
//    leaf.score(scores);
//    assertTrue(abs(scores[0] - 1/3. * log(5)) < 2 * pow(1/3., 4));
//  }
//
//  public void testScores3Classes() {
//    final Vec target = new ArrayVec(
//            0, 0, 1, 1, 1, 2, 2, 2, 2
//    );
//    final Mx data = new VecBasedMx(1, new ArrayVec(
//            1, 1, 1, 0, 0, 1, 1, 1, 0
//    ));
//    final DataSetImpl ds = new DataSetImpl(data, target);
//    final BFGrid grid = GridTools.medianGrid(ds, 2);
//    assertEquals(grid.size(), 1);
//    final BinarizedDataSet bds = new BinarizedDataSet(ds, grid);
//    final MultiLLClassificationLeaf leaf = new MultiLLClassificationLeaf(bds, new Vec[]{
//                                                                            new ArrayVec(target.dim()),
//                                                                            new ArrayVec(target.dim()),
//                                                                            new ArrayVec(target.dim())
//                                                                         }, target, VecTools.fill(new ArrayVec(target.dim()), 1));
//    assertTrue(Arrays.equals(new boolean[]{true, true, true}, leaf.mask()));
//    final double[] scores = new double[1];
//    leaf.score(scores);
//    assertTrue(abs(scores[0] - 2.9375708) < MathTools.EPSILON);
//    assertTrue(leaf.alpha() < 0);
//  }
}
