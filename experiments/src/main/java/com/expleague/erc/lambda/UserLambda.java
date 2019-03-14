package com.expleague.erc.lambda;

import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;

import java.util.HashMap;
import java.util.Map;
import java.lang.Math;

public final class UserLambda {
    private final ArrayVec userEmbedding;
    private final Map<String, ArrayVec> projectEmbeddings;
    private final double beta;
    private final double otherProjectImportance;
    private double currentTime;
    private final Map<String, Double> lastTimeOfProjects;
    private double commonSum;
    private final Map<String, Double> additionalSumByProject;
    private final ArrayVec commonUserDerivative;
    private final Map<String, ArrayVec> userDerivativeByProject;
    private final ArrayVec commonProjectDerivative;
    private final Map<String, ArrayVec> projectDerivativeByProject;
    private final int dimensionality;

    public UserLambda(ArrayVec userEmbedding, Map<String, ArrayVec> projectEmbeddings, double beta, double otherProjectImportance) {
        this.userEmbedding = userEmbedding;
        this.projectEmbeddings = projectEmbeddings;
        this.beta = beta;
        this.otherProjectImportance = otherProjectImportance;
        dimensionality = userEmbedding.dim();

        currentTime = 0.;
        lastTimeOfProjects = new HashMap<>();

        commonSum = 0;
        additionalSumByProject = new HashMap<>();

        commonUserDerivative = new ArrayVec(dimensionality);
        VecTools.fill(commonUserDerivative, 0.);
        userDerivativeByProject = new HashMap<>();

        commonProjectDerivative = new ArrayVec(dimensionality);
        VecTools.fill(commonProjectDerivative, 0);
        projectDerivativeByProject = new HashMap<>();
    }

    public final void update(String projectId, double timeDelta) {
        timeDelta = 1.;
        if (!lastTimeOfProjects.containsKey(projectId)) {
            lastTimeOfProjects.put(projectId, currentTime);
        }
        double e = Math.exp(-beta * timeDelta);
        double decay = Math.exp(-beta * (currentTime - lastTimeOfProjects.get(projectId)));
        double scalarProduct = VecTools.multiply(userEmbedding, projectEmbeddings.get(projectId));

        if (!additionalSumByProject.containsKey(projectId)) {
            additionalSumByProject.put(projectId, 0.);
            ArrayVec projectSpecificProjectDerivative = new ArrayVec(dimensionality);
            VecTools.fill(projectSpecificProjectDerivative, 0.);
            userDerivativeByProject.put(projectId, projectSpecificProjectDerivative);
            ArrayVec projectSpecificUserDerivative = new ArrayVec(dimensionality);
            VecTools.fill(projectSpecificUserDerivative, 0.);
            projectDerivativeByProject.put(projectId, projectSpecificUserDerivative);
        }

        // Updating lambda
        commonSum = e * commonSum + otherProjectImportance * scalarProduct;
        additionalSumByProject.put(projectId,
                decay * additionalSumByProject.get(projectId) + (1 - otherProjectImportance) * scalarProduct);

        // Updating user derivative
        ArrayVec commonUserDerivativeAdd = new ArrayVec();
        commonUserDerivativeAdd.assign(projectEmbeddings.get(projectId));
        commonUserDerivativeAdd.scale(otherProjectImportance);
        commonUserDerivative.scale(e);
        commonUserDerivative.add(commonUserDerivativeAdd);
        ArrayVec projectUserDerivativeAdd = new ArrayVec();
        projectUserDerivativeAdd.assign(projectEmbeddings.get(projectId));
        projectUserDerivativeAdd.scale(1 - otherProjectImportance);
        userDerivativeByProject.get(projectId).scale(decay);
        userDerivativeByProject.get(projectId).add(projectUserDerivativeAdd);

        // Updating project derivative
        ArrayVec commonProjectDerivativeAdd = new ArrayVec(dimensionality);
        commonProjectDerivativeAdd.assign(userEmbedding);
        commonProjectDerivativeAdd.scale(otherProjectImportance);
        commonProjectDerivative.scale(e);
        commonProjectDerivative.add(commonProjectDerivativeAdd);
        ArrayVec projectDerivativeByProjectAdd = new ArrayVec();
        projectDerivativeByProjectAdd.assign(userEmbedding);
        projectDerivativeByProjectAdd.scale(1 - otherProjectImportance);
        projectDerivativeByProject.get(projectId).scale(decay);
        projectDerivativeByProject.get(projectId).add(projectDerivativeByProjectAdd);

        lastTimeOfProjects.put(projectId, currentTime);
        currentTime += timeDelta;
    }

    public final double getLambda(String projectId) {
        if (!additionalSumByProject.containsKey(projectId)) {
            return commonSum + VecTools.multiply(userEmbedding, projectEmbeddings.get(projectId));
        }
        return commonSum + VecTools.multiply(userEmbedding, projectEmbeddings.get(projectId)) +
                additionalSumByProject.get(projectId);
    }

    public final ArrayVec getLambdaUserDerivative(String projectId) {
        ArrayVec completeDerivative = new ArrayVec(dimensionality);
        completeDerivative.assign(commonUserDerivative);
        completeDerivative.add(projectEmbeddings.get(projectId));
        if (userDerivativeByProject.containsKey(projectId)) {
            ArrayVec completeDerivativeAdd = new ArrayVec(dimensionality);
            completeDerivativeAdd.assign(userDerivativeByProject.get(projectId));
            double decay = currentTime - lastTimeOfProjects.get(projectId);
            completeDerivative.scale(decay);
            completeDerivative.add(completeDerivativeAdd);
        }
        return completeDerivative;
    }

    public final Map<String, ArrayVec> getLambdaProjectDerivative(String projectId) {
        Map<String, ArrayVec> derivative = new HashMap<>();
        for (String p: lastTimeOfProjects.keySet()) {
            ArrayVec initialDerivative = new ArrayVec(dimensionality);
            initialDerivative.assign(commonProjectDerivative);
            derivative.put(p, initialDerivative);
        }
        if (projectDerivativeByProject.containsKey(projectId)) {
            double decay = currentTime - lastTimeOfProjects.get(projectId);
            ArrayVec derivativeAdd = new ArrayVec(dimensionality);
            derivativeAdd.assign(projectDerivativeByProject.get(projectId));
            derivativeAdd.scale(decay);
            derivativeAdd.add(userEmbedding);
            derivative.get(projectId).add(derivativeAdd);
        }
        return derivative;
    }
}
