package com.spbsu.ml;

import com.spbsu.commons.math.Func;
import com.spbsu.commons.math.Trans;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.ml.dynamicGrid.impl.BFDynamicGrid;
import com.spbsu.ml.dynamicGrid.interfaces.DynamicGrid;
import com.spbsu.ml.dynamicGrid.models.ObliviousTreeDynamicBin;
import com.spbsu.ml.func.Ensemble;
import com.spbsu.ml.func.FuncEnsemble;
import com.spbsu.ml.func.FuncJoin;
import com.spbsu.ml.io.ModelsSerializationRepository;
import com.spbsu.ml.models.FMModel;
import com.spbsu.ml.models.ObliviousTree;
import com.spbsu.ml.models.Region;
import com.spbsu.ml.models.multiclass.JoinedBinClassModel;
import com.spbsu.ml.models.multiclass.JoinedProbsModel;
import com.spbsu.ml.models.multilabel.MultiLabelBinarizedModel;

import java.util.Arrays;

/**
 * User: solar
 * Date: 12.08.13
 * Time: 14:02
 */
public class SerializationTest extends GridTest {
  protected static BFGrid grid;
  protected static DynamicGrid dynamicGrid;
  @Override
  protected void setUp() throws Exception {
    super.setUp();
    grid = GridTools.medianGrid(learn.vecData(), 32);
    dynamicGrid = new BFDynamicGrid(learn.vecData(), 32);
  }

  public void testObliviousTree() {
    final ObliviousTree ot = new ObliviousTree(Arrays.asList(grid.bf(20)), new double[]{0, 1}, new double[]{10, 3});
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


  public void testRegion() {
    final Region region = new Region(Arrays.asList(grid.bf(20), grid.bf(5), grid.bf(50),grid.bf(22)), new boolean[]{true,false,false,true}, 42,0,3,101.1,1);
    ModelsSerializationRepository serialization = new ModelsSerializationRepository();
    boolean caught = false;
    try {
      serialization.read(serialization.write(region), Region.class);
    }
    catch (RuntimeException re) {
      caught = true;
    }
    assertTrue(caught);
    serialization = new ModelsSerializationRepository(grid);
    assertEquals(region, serialization.read(serialization.write(region), Region.class));
  }

  public void testObliviousTreeDynamicBin() {
    final ObliviousTreeDynamicBin ot = new ObliviousTreeDynamicBin(Arrays.asList(dynamicGrid.bf(0, 2)), new double[]{0, 1});
    ModelsSerializationRepository serialization = new ModelsSerializationRepository();
    boolean caught = false;
    try {
      serialization.read(serialization.write(ot), ObliviousTreeDynamicBin.class);
    } catch (RuntimeException re) {
      caught = true;
    }
    assertTrue(caught);
    serialization = new ModelsSerializationRepository(dynamicGrid);
    assertEquals(ot, serialization.read(serialization.write(ot), ObliviousTreeDynamicBin.class));
  }

  public void testDynamicGrid() {
    final ModelsSerializationRepository serialization = new ModelsSerializationRepository();
    assertEquals(dynamicGrid.toString(), serialization.read(serialization.write(dynamicGrid), DynamicGrid.class).toString());
    assertEquals(dynamicGrid, serialization.read(serialization.write(dynamicGrid), DynamicGrid.class));
  }


  public void testGrid() {
    final ModelsSerializationRepository serialization = new ModelsSerializationRepository();
    assertEquals(grid.toString(), serialization.read(serialization.write(grid), BFGrid.class).toString());
    assertEquals(grid, serialization.read(serialization.write(grid), BFGrid.class));
  }

  public void testAdditiveModel() {
    final ObliviousTree ot = new ObliviousTree(Arrays.asList(grid.bf(20)), new double[]{0, 1}, new double[]{10, 3});
    final Ensemble sum = new Ensemble(Arrays.<Trans>asList(ot, ot, ot), 0.1);
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

  public void testFuncEnsemble() throws Exception {
    final double v = 100500.;
    final FMModel func = new FMModel(new VecBasedMx(1, new ArrayVec(v)));
    final FuncEnsemble<FMModel> funcEnsemble = new FuncEnsemble<>(new FMModel[]{func}, new ArrayVec(0.5));

    final ModelsSerializationRepository repository = new ModelsSerializationRepository();
    final FuncEnsemble readModel = repository.read(repository.write(funcEnsemble), FuncEnsemble.class);
    assertEquals(funcEnsemble, readModel);
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
    final ModelsSerializationRepository serialization = new ModelsSerializationRepository();
    assertEquals(model, serialization.read(serialization.write(model), FMModel.class));
  }

  public void testJoinedBinClassModel() throws Exception {
    final double v = 100500.;
    final FMModel func = new FMModel(new VecBasedMx(1, new ArrayVec(v)));
    final JoinedBinClassModel joinedBinClassModel = new JoinedBinClassModel(new Func[]{func});
    final ModelsSerializationRepository repository = new ModelsSerializationRepository();
    final JoinedBinClassModel readModel = repository.read(repository.write(joinedBinClassModel), JoinedBinClassModel.class);
    assertEquals(joinedBinClassModel, readModel);
  }

  public void testMultiLabelLogitModel() throws Exception {
    final double v = 100500.;
    final FMModel func = new FMModel(new VecBasedMx(1, new ArrayVec(v)));
    final MultiLabelBinarizedModel binarizedModel = new MultiLabelBinarizedModel(new FuncJoin(new Func[]{func}));
    final ModelsSerializationRepository repository = new ModelsSerializationRepository();
    final CharSequence write = repository.write(binarizedModel);
    final MultiLabelBinarizedModel readModel = repository.read(write, MultiLabelBinarizedModel.class);
    assertEquals(binarizedModel, readModel);
  }

  public void testJoinedProbsModel() throws Exception {
    final double v = 100500.;
    final FMModel func = new FMModel(new VecBasedMx(1, new ArrayVec(v)));
    final JoinedProbsModel joinedProbsModel = new JoinedProbsModel(new Func[]{func});
    final ModelsSerializationRepository repository = new ModelsSerializationRepository();
    final JoinedProbsModel readModel = repository.read(repository.write(joinedProbsModel), JoinedProbsModel.class);
    assertEquals(joinedProbsModel, readModel);
  }
}
