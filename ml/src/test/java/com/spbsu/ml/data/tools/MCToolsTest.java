package com.spbsu.ml.data.tools;

import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.seq.IntSeq;
import gnu.trove.list.TDoubleList;
import gnu.trove.list.TIntList;
import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.list.linked.TIntLinkedList;
import gnu.trove.map.TIntObjectMap;
import junit.framework.TestCase;

import java.util.Arrays;

/**
 * User: qdeee
 * Date: 30.07.14
 */
public class MCToolsTest extends TestCase {
  private IntSeq target;

  @Override
  protected void setUp() throws Exception {
    super.setUp();
    final int[] classes = {
        15, 15, 15, 15, 15,
        8, 8, 8, 8,
        3, 3, 3,
        2, 2,
        1, 0};
    target = new IntSeq(classes);
  }

  public void testCountClasses() throws Exception {
    assertEquals(16, MCTools.countClasses(target));
  }

  public void testClassEntriesCount() throws Exception {
    assertEquals(5, MCTools.classEntriesCount(target, 15));
    assertEquals(1, MCTools.classEntriesCount(target, 0));
  }

  public void testExtractClassForBinary() throws Exception {
    final IntSeq binClassTarget = MCTools.extractClassForBinary(target, 8);
    final IntSeq expectedTarget = new IntSeq(new int[]{
        1, 1, 1, 1, 1,
        0, 0, 0, 0,
        1, 1, 1,
        1, 1,
        1, 1
    });
    assertEquals(expectedTarget, binClassTarget);
  }

  public void testGetClassesLabels() throws Exception {
    final int[] expectedLabels = {15, 8, 3, 2, 1, 0};
    assertTrue(Arrays.equals(expectedLabels, MCTools.getClassesLabels(target)));
  }

  public void testNormalizeTarget() throws Exception {
    final TIntArrayList labels = new TIntArrayList();
    final IntSeq normalizedTarget = MCTools.normalizeTarget(target, labels);

    final TIntArrayList expectedLabels = new TIntArrayList(new int[]{15, 8, 3, 2, 1, 0});
    final IntSeq expectedTarget = new IntSeq(new int[]{
        0, 0, 0, 0, 0,
        1, 1, 1, 1,
        2, 2, 2,
        3, 3,
        4, 5
    });
    assertEquals(expectedLabels, labels);
    assertEquals(expectedTarget, normalizedTarget);
  }

  public void testSplitClassesIdxs() throws Exception {
    final TIntObjectMap<TIntList> classesIdxs = MCTools.splitClassesIdxs(target);

    final TIntArrayList expectedForClass15 = new TIntArrayList(new int[]{0, 1, 2, 3, 4});
    assertTrue(Arrays.equals(expectedForClass15.toArray(), classesIdxs.get(15).toArray()));

    final TIntArrayList expectedForClass3 = new TIntArrayList(new int[]{9, 10, 11});
    assertTrue(Arrays.equals(expectedForClass3.toArray(), classesIdxs.get(3).toArray()));
  }

  public void testTransformRegressionToMC() throws Exception {
    final Vec regressionTarget = new ArrayVec(0, 0, 0.33, 0.33, 0.66, 0.66, 1, 1);
    final TDoubleList borders = new TDoubleArrayList();
    final IntSeq mcTarget = MCTools.transformRegressionToMC(regressionTarget, 4, borders);

    final IntSeq expectedMCTarget = new IntSeq(new int[]{0, 0, 1, 1, 2, 2, 3, 3});
    final TDoubleList expectedBorders = new TDoubleArrayList(new double[]{0.25, 0.5, 0.75, 1.0});
    assertEquals(expectedMCTarget, mcTarget);
    assertEquals(expectedBorders, borders);
  }
}
