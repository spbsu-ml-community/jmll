package com.expleague.ml;

import com.expleague.commons.math.Func;
import com.expleague.commons.math.Trans;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.func.Ensemble;

import java.util.function.Consumer;

/**
 * User: solar
 * Date: 26.03.15
 * Time: 18:58
 */
public class ScorePrinter implements Consumer<Trans> {
    private final String message;
    private final Vec cursor;
    private final VecDataSet ds;
    private final Func metric;
    private int index = 0;
    private final int step = 10;

    ScorePrinter(String message, VecDataSet ds, Func metric) {
        this.message = message;
        this.ds = ds;
        this.metric = metric;
        cursor = new ArrayVec(ds.length());
    }

    @Override
    public void accept(Trans partial) {
        if (partial instanceof Ensemble) {
            final Ensemble linear = (Ensemble) partial;
            final Trans increment = linear.last();
            for (int i = 0; i < ds.length(); i++) {
                if (increment instanceof Ensemble) {
                    cursor.adjust(i, linear.wlast() * (increment.trans(ds.data().row(i)).get(0)));
                } else {
                    cursor.adjust(i, linear.wlast() * ((Func) increment).value(ds.data().row(i)));
                }
            }
        } else {
            for (int i = 0; i < ds.length(); i++) {
                cursor.set(i, ((Func) partial).value(ds.data().row(i)));
            }
        }

        if (++index % step == 0) {
//        System.out.println(index);
            System.out.println(index + " " + message + " " + metric.getClass().getSimpleName() + ":" + metric.value(cursor));
        }
    }
}
