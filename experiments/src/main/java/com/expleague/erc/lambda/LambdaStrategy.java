package com.expleague.erc.lambda;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.erc.Event;
import gnu.trove.map.TIntObjectMap;

public interface LambdaStrategy {
    double getLambda(final int userId, final int itemId);

    Vec getLambdaUserDerivative(final int userId, final int itemId);

    TIntObjectMap<Vec> getLambdaItemDerivative(final int userId, final int itemId);

    void accept(final Event event);
}
