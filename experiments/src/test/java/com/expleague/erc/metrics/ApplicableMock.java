package com.expleague.erc.metrics;

import com.expleague.erc.Event;
import com.expleague.erc.Session;
import com.expleague.erc.models.ApplicableModel;

import static java.lang.Math.*;

public class ApplicableMock implements ApplicableModel {
    private final double lambda;

    public ApplicableMock(double lambda) {
        this.lambda = lambda;
    }

    @Override
    public void accept(final Session session) {

    }

    @Override
    public double getLambda(int userId, int itemId) {
        return lambda;
    }

    @Override
    public double timeDelta(int userId, int itemId) {
        return 1 / lambda;
    }

    @Override
    public double probabilityBeforeX(int userId, int itemId, double x) {
        return 1 - exp(-getLambda(userId, itemId) * x);
    }
}
