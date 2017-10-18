package com.expleague.ml;

import com.expleague.ml.dynamicGrid.interfaces.DynamicGrid;

public interface DynamicGridEnabled {
  DynamicGrid getGrid();

  void setGrid(DynamicGrid grid);
}
