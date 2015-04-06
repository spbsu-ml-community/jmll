package com.spbsu.ml.data.cherry;

import com.spbsu.commons.util.BestHolder;
import com.spbsu.ml.BFGrid;

public class CherryBestHolder extends BestHolder<BFGrid.BFRow> {
  private int startBin;
  private int endBin;

  public synchronized boolean update(final BFGrid.BFRow update, final double score, final int startBin, final int endBin) {
    if (update(update, score)) {
      this.startBin = startBin;
      this.endBin = endBin;
      return true;
    }
    return false;
  }

  public int startBin() {
    return startBin;
  }

  public int endBin() {
    return endBin;
  }
}
