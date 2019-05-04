package com.expleague.erc.lambda;

import com.expleague.commons.math.vectors.Vec;
import gnu.trove.map.TIntObjectMap;

public interface UserLambda {
    void reset();

    void update(final int itemId, double timeDelta);

    double getLambda(final int itemId);

    Vec getLambdaUserDerivative(final int itemId);

    TIntObjectMap<Vec> getLambdaItemsDerivative(final int itemId);
}
