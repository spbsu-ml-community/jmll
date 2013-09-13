package com.spbsu.ml.io;

import com.spbsu.commons.filters.Filter;
import com.spbsu.commons.func.types.ConversionRepository;
import com.spbsu.commons.func.types.SerializationRepository;
import com.spbsu.commons.func.types.TypeConverter;
import com.spbsu.commons.func.types.impl.TypeConvertersCollection;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.GridEnabled;

/**
 * User: solar
 * Date: 12.08.13
 * Time: 13:01
 */
public class ModelsSerializationRepository extends SerializationRepository<CharSequence> {
  private static ConversionRepository conversion = new TypeConvertersCollection(new ObliviousTreeConversionPack(),
                                                                                new ObliviousMultiClassTreeConversionPack(),
                                                                                new AdditiveModelConversionPack(),
                                                                                new AdditiveMultiClassModelConversionPack(),
                                                                                BFGrid.CONVERTER.getClass());
  private BFGrid grid;

  public ModelsSerializationRepository() {
    super(conversion, CharSequence.class);
  }

  public ModelsSerializationRepository(final BFGrid grid) {
    super(conversion.customize(new Filter<TypeConverter>() {
      @Override
      public boolean accept(TypeConverter typeConverter) {
        if (typeConverter instanceof GridEnabled)
          ((GridEnabled) typeConverter).setGrid(grid);
        return true;
      }
    }), CharSequence.class);
  }

  private ModelsSerializationRepository(ConversionRepository repository) {
    super(repository, CharSequence.class);
  }

  public ModelsSerializationRepository customizeGrid(final BFGrid grid) {
    return new ModelsSerializationRepository(base.customize(new Filter<TypeConverter>() {
      @Override
      public boolean accept(TypeConverter typeConverter) {
        if (typeConverter instanceof GridEnabled)
          ((GridEnabled) typeConverter).setGrid(grid);
        return true;
      }
    }));
  }
}
