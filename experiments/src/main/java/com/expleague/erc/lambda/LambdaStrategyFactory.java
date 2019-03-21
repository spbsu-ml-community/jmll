package com.expleague.erc.lambda;

import com.expleague.commons.math.vectors.Vec;
import gnu.trove.map.TIntObjectMap;

public interface LambdaStrategyFactory {
    LambdaStrategy get(final TIntObjectMap<Vec> userEmbeddings, final TIntObjectMap<Vec> itemEmbeddings,
                       final double beta, final double otherProjectImportance);
}
