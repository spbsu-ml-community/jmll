package com.expleague.expedia.utils;

public class CheckPoint {
  private final int total;
  private final int step;
  private int current;

  private long start = System.currentTimeMillis();

  public CheckPoint(final int step) {
    this(0, step);
  }

  public CheckPoint(final int total, final int step) {
    this.total = total;
    this.step = step;
    this.current = 0;
  }

  public boolean check() {
    ++current;

    if (current % step == 0 || current == total) {
      final long duration = (System.currentTimeMillis() - start) / 1_000;
      final long memoryUsage = (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / 1_000_000;
      System.out.println(String.format("Check point:\tstep: %d\ttime: %ds\tMemory used: %dMb", current, duration, memoryUsage));
      start = System.currentTimeMillis();

      return true;
    }

    return false;
  }
}
