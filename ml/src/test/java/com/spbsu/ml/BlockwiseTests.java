package com.spbsu.ml;

import com.spbsu.commons.math.vectors.MxTools;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.seq.IntSeq;
import com.spbsu.commons.util.Pair;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.tools.MCTools;
import com.spbsu.ml.func.Ensemble;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.MLLLogit;
import com.spbsu.ml.loss.SatL2;
import com.spbsu.ml.loss.blockwise.BlockwiseL2;
import com.spbsu.ml.loss.blockwise.BlockwiseMLLLogit;
import com.spbsu.ml.loss.blockwise.BlockwiseSatL2;
import com.spbsu.ml.loss.blockwise.BlockwiseWeightedLoss;
import com.spbsu.ml.loss.multiclass.MCMacroPrecision;
import com.spbsu.ml.methods.GradientBoosting;
import com.spbsu.ml.methods.MultiClass;
import com.spbsu.ml.methods.trees.GreedyObliviousTree;
import com.spbsu.ml.models.MultiClassModel;
import gnu.trove.list.array.TDoubleArrayList;
import junit.framework.TestCase;

import java.io.IOException;

/**
* User: qdeee
* Date: 16.07.14
*/
public class BlockwiseTests extends TestCase {
  private IntSeq intTarget;
  private Vec doubleTarget;
  private Vec point;

  @Override
  protected void setUp() throws Exception {
    super.setUp();

    final int poolSize = 50;

    point = new ArrayVec(poolSize);
    intTarget = new IntSeq(new int[poolSize]);
    doubleTarget = new ArrayVec(poolSize);

    for (int i = 0; i < poolSize; i++) {
      point.set(i, Double.parseDouble("." + i));
      intTarget.arr[i] = i % 4;
      doubleTarget.set(i, Double.parseDouble("." + (poolSize - i - 1)));
    }
  }

  public void testMulticlass() throws IOException {
    final TDoubleArrayList borders = new TDoubleArrayList(new double[]{0.038125, 0.07625, 0.114375, 0.1525, 0.61});
    final Pair<VecDataSet,IntSeq> pair = MCTools.loadRegressionAsMC("./jmll/ml/src/test/data/features.txt.gz", 5, borders);

    VecDataSet ds = pair.first;
    final IntSeq intTarget = pair.second;

    BlockwiseMLLLogit newTarget = new BlockwiseMLLLogit(intTarget, ds);
    MLLLogit oldTarget = new MLLLogit(intTarget, ds);

    final BFGrid grid = GridTools.medianGrid(ds, 32);

    final GradientBoosting<BlockwiseMLLLogit> boosting = new GradientBoosting<>(
        new MultiClass(new GreedyObliviousTree(grid, 5), SatL2.class),
        20, 0.5);
    final Ensemble ensemble = boosting.fit(ds, newTarget);
    final MultiClassModel model = MultiClassModel.joinBoostingResults(ensemble);
    final Func mcMacroPrecision = new MCMacroPrecision(newTarget.labels(), ds);
    final double value = mcMacroPrecision.value(model.bestClassAll(ds.data()));
    System.out.println(value);
  }

  public void testMLLLogit() throws Exception {
    final MLLLogit oldMLLLogit = new MLLLogit(intTarget, null);
    final BlockwiseMLLLogit newMLLLogit = new BlockwiseMLLLogit(intTarget, null);

    final int poolSize = intTarget.length();
    final Vec point1 = new ArrayVec(oldMLLLogit.dim());
    for (int i = 0; i < point1.dim(); i++) {
      point1.set(i, Double.parseDouble("." + i));
    }
    final Vec point2 = MxTools.transpose(new VecBasedMx(poolSize, point1));

    assertEquals(oldMLLLogit.value(point1), newMLLLogit.value(point2), 1e-15);

    final Vec oldGrad = oldMLLLogit.gradient(point1);
    final Vec newGrad = newMLLLogit.gradient(point2);
    final VecBasedMx oldGradMx = new VecBasedMx(poolSize, oldGrad);
    final VecBasedMx newGradMx = new VecBasedMx(newMLLLogit.classesCount() - 1, newGrad);
    assertEquals(MxTools.transpose(oldGradMx), newGradMx);
  }

  public void testWeightedLoss() throws Exception {
    final int[] weights = new int[intTarget.length()];
    for (int i = 0; i < weights.length; i++) {
      weights[i] = 1;
    }

    final BlockwiseSatL2 satL2 = new BlockwiseSatL2(doubleTarget, null);
    final BlockwiseWeightedLoss<BlockwiseSatL2> weightedLoss = new BlockwiseWeightedLoss<>(satL2, weights);

    final Vec gradient1 = satL2.gradient(point);
    final Vec gradient2 = weightedLoss.gradient(point);
    assertEquals(gradient1, gradient2);

    assertEquals(satL2.value(point), weightedLoss.value(point), 1e-15);
  }

  public void testL2() throws Exception {
    final L2 oldL2 = new L2(doubleTarget, null);
    final BlockwiseL2 newL2 = new BlockwiseL2(doubleTarget, null);
    assertEquals(oldL2.gradient(point), newL2.gradient(point));
    assertEquals(oldL2.value(point), newL2.value(point), 1e-15);
  }
}
