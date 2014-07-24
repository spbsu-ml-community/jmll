package com.spbsu.ml.DynamicGrid.Interface;



import java.util.ArrayList;

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

    void setOwner(DynamicGrid grid);

    int bin(double v);

}
