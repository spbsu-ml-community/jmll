package com.expleague.erc.lambda;

import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;

import java.util.Map;

public interface LambdaStrategyFactory {
    LambdaStrategy get(Map<String, ArrayVec> userEmbeddings, Map<String, ArrayVec> itemEmbeddings, double beta,
                       double otherProjectImportance);
}
