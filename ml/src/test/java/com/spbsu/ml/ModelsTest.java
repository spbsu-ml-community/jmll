package com.spbsu.ml;

import com.spbsu.commons.math.Func;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.ml.func.FuncEnsemble;
import com.spbsu.ml.models.ModelTools;
import com.spbsu.ml.models.ObliviousTree;
import junit.framework.TestCase;

import java.util.Arrays;

/**
 * User: solar
 * Date: 28.04.14
 * Time: 13:18
 */
public class ModelsTest extends TestCase {
  public void testOTlvl2Compile() {
    final BFGrid grid = new BFGrid(new BFGrid.BFRow[]{
            new BFGrid.BFRow(0, 0, new double[]{1, 2, 3, 4, 5}),
            new BFGrid.BFRow(5, 1, new double[]{1, 2, 3, 4, 5}),
    });
    final FuncEnsemble<ObliviousTree> sample = new FuncEnsemble<ObliviousTree>(Arrays.asList(new ObliviousTree(
            Arrays.asList(grid.row(0).bf(0), grid.row(1).bf(0)),
            new double[]{1, 2, 3, 4}
    )), 1);

    assertEquals(1., sample.value(new ArrayVec(1, 1)));
    assertEquals(2., sample.value(new ArrayVec(1, 2)));
    assertEquals(3., sample.value(new ArrayVec(2, 1)));
    assertEquals(4., sample.value(new ArrayVec(2, 2)));

    final Func compile = ModelTools.compile(sample);

    assertEquals(1., compile.value(new ArrayVec(1, 1)));
    assertEquals(2., compile.value(new ArrayVec(1, 2)));
    assertEquals(3., compile.value(new ArrayVec(2, 1)));
    assertEquals(4., compile.value(new ArrayVec(2, 2)));
  }

  public void testOTlvl3Compile() {
    final BFGrid grid = new BFGrid(new BFGrid.BFRow[]{
            new BFGrid.BFRow(0, 0, new double[]{1}),
            new BFGrid.BFRow(1, 1, new double[]{1}),
            new BFGrid.BFRow(2, 2, new double[]{1}),
    });
    final FuncEnsemble<ObliviousTree> sample = new FuncEnsemble<ObliviousTree>(Arrays.asList(new ObliviousTree(
            Arrays.asList(grid.row(0).bf(0), grid.row(1).bf(0), grid.row(2).bf(0)),
            new double[]{1, 2, 3, 4, 5, 6, 7, 8}
    )), 1);

    assertEquals(1., sample.value(new ArrayVec(1, 1, 1)));
    assertEquals(2., sample.value(new ArrayVec(1, 1, 2)));
    assertEquals(3., sample.value(new ArrayVec(1, 2, 1)));
    assertEquals(4., sample.value(new ArrayVec(1, 2, 2)));
    assertEquals(5., sample.value(new ArrayVec(2, 1, 1)));
    assertEquals(6., sample.value(new ArrayVec(2, 1, 2)));
    assertEquals(7., sample.value(new ArrayVec(2, 2, 1)));
    assertEquals(8., sample.value(new ArrayVec(2, 2, 2)));

    final Func compile = ModelTools.compile(sample);

    assertEquals(1., compile.value(new ArrayVec(1, 1, 1)));
    assertEquals(2., compile.value(new ArrayVec(1, 1, 2)));
    assertEquals(3., compile.value(new ArrayVec(1, 2, 1)));
    assertEquals(4., compile.value(new ArrayVec(1, 2, 2)));
    assertEquals(5., compile.value(new ArrayVec(2, 1, 1)));
    assertEquals(6., compile.value(new ArrayVec(2, 1, 2)));
    assertEquals(7., compile.value(new ArrayVec(2, 2, 1)));
    assertEquals(8., compile.value(new ArrayVec(2, 2, 2)));
  }

  public void testOTlvl4Compile() {
    final BFGrid grid = new BFGrid(new BFGrid.BFRow[]{
            new BFGrid.BFRow(0, 0, new double[]{1}),
            new BFGrid.BFRow(1, 1, new double[]{1}),
            new BFGrid.BFRow(2, 2, new double[]{1}),
            new BFGrid.BFRow(3, 3, new double[]{1}),
    });
    final FuncEnsemble<ObliviousTree> sample = new FuncEnsemble<ObliviousTree>(Arrays.asList(new ObliviousTree(
            Arrays.asList(
                    grid.row(0).bf(0),
                    grid.row(1).bf(0),
                    grid.row(2).bf(0),
                    grid.row(3).bf(0)),
            new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
    )), 1);

    assertEquals(1., sample.value(new ArrayVec(1, 1, 1, 1)));
    assertEquals(2., sample.value(new ArrayVec(1, 1, 1, 2)));
    assertEquals(3., sample.value(new ArrayVec(1, 1, 2, 1)));
    assertEquals(4., sample.value(new ArrayVec(1, 1, 2, 2)));
    assertEquals(5., sample.value(new ArrayVec(1, 2, 1, 1)));
    assertEquals(6., sample.value(new ArrayVec(1, 2, 1, 2)));
    assertEquals(7., sample.value(new ArrayVec(1, 2, 2, 1)));
    assertEquals(8., sample.value(new ArrayVec(1, 2, 2, 2)));
    assertEquals(9., sample.value(new ArrayVec(2, 1, 1, 1)));
    assertEquals(10., sample.value(new ArrayVec(2, 1, 1, 2)));
    assertEquals(11., sample.value(new ArrayVec(2, 1, 2, 1)));
    assertEquals(12., sample.value(new ArrayVec(2, 1, 2, 2)));
    assertEquals(13., sample.value(new ArrayVec(2, 2, 1, 1)));
    assertEquals(14., sample.value(new ArrayVec(2, 2, 1, 2)));
    assertEquals(15., sample.value(new ArrayVec(2, 2, 2, 1)));
    assertEquals(16., sample.value(new ArrayVec(2, 2, 2, 2)));

    final Func compile = ModelTools.compile(sample);

    assertEquals(1., compile.value(new ArrayVec(1, 1, 1, 1)));
    assertEquals(2., compile.value(new ArrayVec(1, 1, 1, 2)));
    assertEquals(3., compile.value(new ArrayVec(1, 1, 2, 1)));
    assertEquals(4., compile.value(new ArrayVec(1, 1, 2, 2)));
    assertEquals(5., compile.value(new ArrayVec(1, 2, 1, 1)));
    assertEquals(6., compile.value(new ArrayVec(1, 2, 1, 2)));
    assertEquals(7., compile.value(new ArrayVec(1, 2, 2, 1)));
    assertEquals(8., compile.value(new ArrayVec(1, 2, 2, 2)));
    assertEquals(9., compile.value(new ArrayVec(2, 1, 1, 1)));
    assertEquals(10., compile.value(new ArrayVec(2, 1, 1, 2)));
    assertEquals(11., compile.value(new ArrayVec(2, 1, 2, 1)));
    assertEquals(12., compile.value(new ArrayVec(2, 1, 2, 2)));
    assertEquals(13., compile.value(new ArrayVec(2, 2, 1, 1)));
    assertEquals(14., compile.value(new ArrayVec(2, 2, 1, 2)));
    assertEquals(15., compile.value(new ArrayVec(2, 2, 2, 1)));
    assertEquals(16., compile.value(new ArrayVec(2, 2, 2, 2)));
  }
}
