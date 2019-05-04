package com.expleague.erc.lambda;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.erc.Session;
import gnu.trove.map.TIntObjectMap;

import java.io.Serializable;

public interface LambdaStrategy extends Serializable {
    double getLambda(final int userId, final int itemId);

    Vec getLambdaUserDerivative(final int userId, final int itemId);

    TIntObjectMap<Vec> getLambdaItemDerivative(final int userId, final int itemId);

    void accept(final Session session);
}
