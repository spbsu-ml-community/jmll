package com.spbsu.ml;

import com.spbsu.ml.io.ModelsSerializationRepository;
import com.spbsu.ml.func.Linear;
import com.spbsu.ml.models.Ensemble;
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
    grid = GridTools.medianGrid(learn, 32);
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
    Ensemble sum = new Ensemble(Arrays.<Func>asList(ot, ot, ot), 0.1);
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
    assertEquals(sum, serialization.read(serialization.write(sum), Linear.class));
  }

}
