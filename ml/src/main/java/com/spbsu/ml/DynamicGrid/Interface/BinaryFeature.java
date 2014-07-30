package com.spbsu.ml.DynamicGrid.Interface;

import com.spbsu.commons.math.vectors.Vec;

/**
 * Created by noxoomo on 23/07/14.
 */
public interface BinaryFeature {
    public boolean value(Vec vec);

    public DynamicRow row();

    public int binNo();

    public int fIndex();


    public boolean isActive();

    public void setActive(boolean status);

    public double regularization();

    public int useCount();
    public void use();
}
