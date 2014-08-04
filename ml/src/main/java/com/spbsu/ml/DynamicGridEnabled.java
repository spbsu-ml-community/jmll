package com.spbsu.ml;

import com.spbsu.ml.dynamicGrid.interfaces.DynamicGrid;

public interface DynamicGridEnabled {
  DynamicGrid getGrid();

  void setGrid(DynamicGrid grid);
}
