package com.spbsu.ml.dynamicGrid.interfaces;


/**
 * Created by noxoomo on 23/07/14.
 */
public interface DynamicRow {

  public int origFIndex();

  public int size();

  public DynamicGrid grid();

  public boolean addSplit();

  public boolean empty();

  public BinaryFeature bf(int binNo);

  public void setOwner(DynamicGrid grid);

  public short bin(double v);
//    void setActive(BinaryFeature feature);
}
