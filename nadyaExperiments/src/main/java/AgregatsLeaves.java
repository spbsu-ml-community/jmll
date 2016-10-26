import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.data.set.VecDataSet;
import gnu.trove.list.array.TIntArrayList;

/**
 * Created by n_buga on 17.10.16.
 */
public class AgregatsLeaves {

    int featureNo;

    private double[] sumTargets;
    private double[] sumSqrtTargets;
    private int[] countTarget;

    public AgregatsLeaves(Vec targets, VecDataSet dataSet, TIntArrayList marks, int featureNo, int countLeaves) {

        sumTargets = new double[countLeaves];
        sumSqrtTargets = new double[countLeaves];
        countTarget = new int[countLeaves];

        this.featureNo = featureNo;
        int[] ordered = dataSet.order(featureNo);
        for (int i = 0; i < dataSet.length(); i++) {
            sumTargets[marks.get(ordered[i])] += targets.get(ordered[i]);
            sumSqrtTargets[marks.get(ordered[i])] += Math.pow(targets.get(ordered[i]), 2);
            countTarget[marks.get(ordered[i])] += 1;
        }
    }

    public int getFeatureNo() {
        return featureNo;
    }

    public double getSumObserves(int leafID) {
        return sumTargets[leafID];
    }

    public double getSumSquareObserves(int leafID) {
        return sumSqrtTargets[leafID];
    }

    public int getCount(int leafID) {
        return countTarget[leafID];
    }

}
