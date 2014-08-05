package com.spbsu.ml;

import com.spbsu.ml.dynamicGridFix.interfacesFix.DynamicGrid;

public interface DynamicGridEnabled {
  DynamicGrid getGrid();

  void setGrid(DynamicGrid grid);
}
