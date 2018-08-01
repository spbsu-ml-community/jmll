package com.expleague.ml.data.cherry;

import com.expleague.commons.util.BestHolder;
import com.expleague.ml.BFGrid;

public class CherryBestHolder extends BestHolder<BFGrid.Row> {
  private int startBin;
  private int endBin;

  public synchronized boolean update(final BFGrid.Row update, final double score, final int startBin, final int endBin) {
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
