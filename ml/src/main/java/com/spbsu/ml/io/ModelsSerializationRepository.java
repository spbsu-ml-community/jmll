package com.spbsu.ml.io;

import com.spbsu.commons.filters.Filter;
import com.spbsu.commons.func.types.ConversionRepository;
import com.spbsu.commons.func.types.SerializationRepository;
import com.spbsu.commons.func.types.TypeConverter;
import com.spbsu.commons.func.types.impl.TypeConvertersCollection;
import com.spbsu.commons.math.MathTools;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.DynamicGridEnabled;
import com.spbsu.ml.GridEnabled;
import com.spbsu.ml.dynamicGrid.interfaces.DynamicGrid;
import org.jetbrains.annotations.Nullable;

/**
 * User: solar
 * Date: 12.08.13
 * Time: 13:01
 */
public class ModelsSerializationRepository extends SerializationRepository<CharSequence> {
  private static final ConversionRepository conversion = new TypeConvertersCollection(MathTools.CONVERSION,
          new ObliviousTreeConversionPack(),
          new RegionConversionPack(),
          new ObliviousMultiClassTreeConversionPack(),
          new EnsembleModelConversionPack(),
          new FuncEnsembleConversionPack(),
          new TransJoinConversionPack(),
          new FuncJoinConversionPack(),
          new JoinedProbsModelConversionPack(),
          new FMModelConversionPack(),
          new MultiClassModelConversionPack(),
          new JoinedBinClassModelConversionPack(),
          new MultiLabelBinarizedModelConversionPack(),
          new ObliviousTreeDynamicBinConversionPack(),
          BFGrid.CONVERTER.getClass(),
          (new DynamicGridStringConverter()).getClass()
  );
  private BFGrid grid;
  private DynamicGrid dynamicGrid;

  public ModelsSerializationRepository() {
    super(conversion, CharSequence.class);
  }

  public ModelsSerializationRepository(final BFGrid grid) {
    super(conversion.customize(new Filter<TypeConverter>() {
      @Override
      public boolean accept(final TypeConverter typeConverter) {
        if (typeConverter instanceof GridEnabled)
          ((GridEnabled) typeConverter).setGrid(grid);
        return true;
      }
    }), CharSequence.class);
    this.grid = grid;
  }

  public ModelsSerializationRepository(final DynamicGrid grid) {
    super(conversion.customize(new Filter<TypeConverter>() {
      @Override
      public boolean accept(final TypeConverter typeConverter) {
        if (typeConverter instanceof DynamicGridEnabled)
          ((DynamicGridEnabled) typeConverter).setGrid(grid);
        return true;
      }
    }), CharSequence.class);
    this.dynamicGrid = dynamicGrid;
  }


  private ModelsSerializationRepository(final ConversionRepository repository) {
    super(repository, CharSequence.class);
  }

  @Nullable
  public DynamicGrid getDynamicGrid() {
    return dynamicGrid;
  }

  @Nullable
  public BFGrid getGrid() {
    return grid;
  }

  public ModelsSerializationRepository customizeGrid(final BFGrid grid) {
    final ModelsSerializationRepository repository = new ModelsSerializationRepository(base.customize(new Filter<TypeConverter>() {
      @Override
      public boolean accept(final TypeConverter typeConverter) {
        if (typeConverter instanceof GridEnabled)
          ((GridEnabled) typeConverter).setGrid(grid);
        return true;
      }
    }));
    repository.grid = grid;
    return repository;
  }

  public ModelsSerializationRepository customizeGrid(final DynamicGrid grid) {
    final ModelsSerializationRepository repository = new ModelsSerializationRepository(base.customize(new Filter<TypeConverter>() {
      @Override
      public boolean accept(final TypeConverter typeConverter) {
        if (typeConverter instanceof DynamicGridEnabled)
          ((DynamicGridEnabled) typeConverter).setGrid(dynamicGrid);
        return true;
      }
    }));
    repository.dynamicGrid = dynamicGrid;
    return repository;
  }

}
