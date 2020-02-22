package com.expleague.ml.io;

import com.expleague.commons.func.types.ConversionRepository;
import com.expleague.commons.func.types.SerializationRepository;
import com.expleague.commons.math.MathTools;
import com.expleague.commons.func.types.impl.TypeConvertersCollection;
import com.expleague.ml.DynamicGridEnabled;
import com.expleague.ml.GridEnabled;
import com.expleague.ml.dynamicGrid.interfaces.DynamicGrid;
import com.expleague.ml.BFGrid;
import org.jetbrains.annotations.Nullable;

/**
 * User: solar
 * Date: 12.08.13
 * Time: 13:01
 */
public class ModelsSerializationRepository extends SerializationRepository<CharSequence> {
  private static final ConversionRepository conversion = new TypeConvertersCollection(
      MathTools.CONVERSION,
      ModelsSerializationRepository.class,
      ModelsSerializationRepository.class.getPackage().getName()
  );

  private BFGrid grid;
  private DynamicGrid dynamicGrid;

  public ModelsSerializationRepository() {
    super(conversion, CharSequence.class);
  }

  public ModelsSerializationRepository(final BFGrid grid) {
    super(conversion.customize(typeConverter -> {
      if (typeConverter instanceof GridEnabled)
        ((GridEnabled) typeConverter).setGrid(grid);
      return true;
    }), CharSequence.class);
    this.grid = grid;
  }

  public ModelsSerializationRepository(final DynamicGrid grid) {
    super(conversion.customize(typeConverter -> {
      if (typeConverter instanceof DynamicGridEnabled)
        ((DynamicGridEnabled) typeConverter).setGrid(grid);
      return true;
    }), CharSequence.class);
    this.dynamicGrid = grid;
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
    final ModelsSerializationRepository repository = new ModelsSerializationRepository(base.customize(typeConverter -> {
      if (typeConverter instanceof GridEnabled)
        ((GridEnabled) typeConverter).setGrid(grid);
      return true;
    }));
    repository.grid = grid;
    return repository;
  }
}
