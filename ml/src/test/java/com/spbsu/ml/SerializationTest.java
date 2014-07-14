package com.spbsu.ml;

import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.ml.io.ModelsSerializationRepository;
import com.spbsu.ml.func.Linear;
import com.spbsu.ml.func.Ensemble;
import com.spbsu.ml.models.FMModel;
import com.spbsu.ml.models.ObliviousTree;

import java.util.Arrays;

/**
 * User: solar
 * Date: 12.08.13
 * Time: 14:02
 */
public class SerializationTest extends GridTest {
  protected static BFGrid grid;
  @Override
  protected void setUp() throws Exception {
    super.setUp();
    grid = GridTools.medianGrid(learn.vecData(), 32);
  }

  public void testObliviousTree() {
    ObliviousTree ot = new ObliviousTree(Arrays.asList(grid.bf(20)), new double[]{0, 1}, new double[]{10, 3});
    ModelsSerializationRepository serialization = new ModelsSerializationRepository();
    boolean caught = false;
    try {
      serialization.read(serialization.write(ot), ObliviousTree.class);
    }
    catch (RuntimeException re) {
      caught = true;
    }
    assertTrue(caught);
    serialization = new ModelsSerializationRepository(grid);
    assertEquals(ot, serialization.read(serialization.write(ot), ObliviousTree.class));
  }

  public void testGrid() {
    ModelsSerializationRepository serialization = new ModelsSerializationRepository();
    assertEquals(grid.toString(), serialization.read(serialization.write(grid), BFGrid.class).toString());
    assertEquals(grid, serialization.read(serialization.write(grid), BFGrid.class));
  }

  public void testAdditiveModel() {
    ObliviousTree ot = new ObliviousTree(Arrays.asList(grid.bf(20)), new double[]{0, 1}, new double[]{10, 3});
    Ensemble sum = new Ensemble(Arrays.<Trans>asList(ot, ot, ot), 0.1);
    ModelsSerializationRepository serialization = new ModelsSerializationRepository();
    boolean caught = false;
    try {
      serialization.read(serialization.write(sum), ObliviousTree.class);
    }
    catch (RuntimeException re) {
      caught = true;
    }
    assertTrue(caught);
    serialization = new ModelsSerializationRepository(grid);
    assertEquals(sum, serialization.read(serialization.write(sum), Ensemble.class));
  }

  public void testFMModel() {
    final FMModel model = new FMModel(
        new VecBasedMx(
            3,
            new ArrayVec(
                1, 2, 4,
                0, -2.1, 3
            )
        ),
        new ArrayVec(
            0.1, 0.2, 0.3
        ),
        0.5
    );
    ModelsSerializationRepository serialization = new ModelsSerializationRepository();
    assertEquals(model, serialization.read(serialization.write(model), FMModel.class));
  }
}
