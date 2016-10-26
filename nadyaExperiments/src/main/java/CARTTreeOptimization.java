import com.spbsu.commons.math.Trans;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.methods.VecOptimization;
import gnu.trove.list.array.TIntArrayList;

import java.util.Arrays;

/**
 * Created by n_buga on 16.10.16.
 */
public class CARTTreeOptimization<Loss extends L2> extends VecOptimization.Stub<Loss> {

    private double maxError = 0.6;
    private final double EPS = 10e-5;

    private TIntArrayList ownerLeafOfData;
    private VecDataSet learn;
    private Loss loss;

    public CARTTreeOptimization() {}

    public CARTTreeOptimization(double maxError) {
        this.maxError = maxError;
    }

    public double getMaxError() {
        return maxError;
    }

    public void setMaxError(double maxError) {
        this.maxError = maxError;
    }

    public Trans fit(VecDataSet learn, Loss loss) {

        this.loss = loss;
        this.learn = learn;

        CARTTree tree = new CARTTree();
        Leaf firstLeaf = new Leaf();

        ownerLeafOfData = new TIntArrayList(learn.length());
        for (int i = 0; i < learn.length(); i++) {
            ownerLeafOfData.add(firstLeaf.getID());
        }

        setValueAndError(firstLeaf);
        tree.add(firstLeaf);

        if (firstLeaf.getError() - maxError < EPS) {
            return tree;
        }

        constructTree(tree);

        return tree;
    }

    private void setValueAndError(Leaf leaf) {
        int leafID = leaf.getID();
        double sum = 0;
        int count = 0;
        for (int i = 0; i < learn.length(); i++) {
            if (ownerLeafOfData.get(i) == leafID) {
                sum += loss.target().get(i);
                count++;
            }
        }
        double mean = sum/count;
        double error = 0;
        for (int i = 0; i < learn.length(); i++) {
            if (ownerLeafOfData.get(i) == leafID) {
                error += Math.pow(loss.target().get(i) - mean, 2);
            }
        }
        leaf.setValue(mean);
        leaf.setError(error);
    }

    private void constructTree(CARTTree tree) {
        int count = 0;
        int old_size = tree.getLeaves().size();
        while (makeStep(tree) > maxError && count < 100) {
            count++;
            if (old_size == tree.getLeaves().size()) {
                break;
            }
            old_size = tree.getLeaves().size();
        }
    }

    private double makeStep(CARTTree tree) { //return maxError along new leaves

        Condition[] bestCondLeaf = new Condition[tree.getLeaves().size()];
        for (int i = 0; i < tree.getLeaves().size(); i++) {
            bestCondLeaf[i] = new Condition();
        }
        double[] bestError = new double[tree.getLeaves().size()];
        double[] bestErrorLeft = new double[tree.getLeaves().size()];
        double[] bestErrorRight = new double[tree.getLeaves().size()];

        Arrays.fill(bestError, Double.MAX_VALUE);
        double[] bestPartSum = new double[tree.getLeaves().size()];
        int[] bestCount = new int[tree.getLeaves().size()];

        for (int i = 0; i < learn.xdim(); i++) { // sort out feature

            AgregatsLeaves agregat = new AgregatsLeaves(loss.target(), learn, ownerLeafOfData, i, tree.getLeaves().size());

            int[] order = learn.order(i);
            double[] curErrorLeft = new double[tree.getLeaves().size()];
            double[] curErrorRight = new double[tree.getLeaves().size()];
            double[] partSum = new double[tree.getLeaves().size()];
            double[] partSqrSum = new double[tree.getLeaves().size()];
            int[] count = new int[tree.getLeaves().size()];
            double[] last = new double[tree.getLeaves().size()];

            for (int j = 0; j < learn.length(); j++) { //sort out vector on barrier
                int curIndex = order[j];                  //check error of this barrier
                int curLeafID = ownerLeafOfData.get(curIndex);
                if (tree.getLeaves().get(curLeafID).getError() - maxError < EPS) { //if leaf is ok, then do nothing
                    continue;
                }

                if (count[curLeafID] > 0 && last[curLeafID] != learn.data().get(curIndex, i)) { // catch boarder
                    if (curErrorLeft[curLeafID] + curErrorRight[curLeafID] < bestError[curLeafID]) {
                        bestError[curLeafID] = curErrorLeft[curLeafID] + curErrorRight[curLeafID];
                        bestErrorLeft[curLeafID] = curErrorLeft[curLeafID];
                        bestErrorRight[curLeafID] = curErrorRight[curLeafID];
                        bestCondLeaf[curLeafID].set(i, learn.at(order[j]).at(i), true);
                        bestPartSum[curLeafID] = partSum[curLeafID];
                        bestCount[curLeafID] = count[curLeafID];
                    }
                }

                partSum[curLeafID] += loss.target().get(curIndex);
                partSqrSum[curLeafID] += Math.pow(loss.target().get(curIndex), 2);
                count[curLeafID]++;
                curErrorLeft[curLeafID] = partSqrSum[curLeafID] - Math.pow(partSum[curLeafID], 2)/count[curLeafID];
                curErrorRight[curLeafID] = (agregat.getSumSquareObserves(curLeafID) - partSqrSum[curLeafID]) -
                        Math.pow(agregat.getSumObserves(curLeafID) - partSum[curLeafID], 2) /
                                (agregat.getCount(curLeafID) - count[curLeafID]);
                last[curLeafID] = learn.data().get(curIndex, i); //last value of data in this leaf
            }
        }

        double maxErr = 0; //the return value
        int countLeavesBefore = tree.getLeaves().size();
        int pairLeaf[] = new int[countLeavesBefore]; // for recalculate ownerLeafOfData

        for (int i = 0; i < countLeavesBefore; i++) {

            if (tree.getLeaves().get(i).getError() - bestError[i] < EPS) { // if new error worser then old
                maxErr = Math.max(maxErr, tree.getLeaves().get(i).getError());
                pairLeaf[i] = i;
                continue;
            }

            Condition condition1 = bestCondLeaf[i];  // smth < boarder
            Condition condition2 = (new Condition(condition1)).set(false); // smth >= boarder

            AgregatsLeaves curAgregates = new AgregatsLeaves(loss.target(), learn, ownerLeafOfData, condition1.getFeatureNo(),
                    tree.getLeaves().size());

            Leaf newLeaf = new Leaf(tree.getLeaves().get(i)); // copy content of old leaf
            pairLeaf[i] = newLeaf.getID();
            newLeaf.setValue((curAgregates.getSumObserves(i) - bestPartSum[i]) /
                    (curAgregates.getCount(i) - bestCount[i]));
            newLeaf.getListFeatures().addFeature(condition2);
            newLeaf.setError(bestErrorRight[i]);
            tree.add(newLeaf);

            tree.getLeaves().get(i).setValue(bestPartSum[i] / bestCount[i]);
            tree.getLeaves().get(i).getListFeatures().addFeature(condition1);
            tree.getLeaves().get(i).setError(bestErrorLeft[i]);

            maxErr = Math.max(bestErrorLeft[i], Math.max(bestErrorRight[i], maxErr));
        }

        for (int i = 0; i < learn.length(); i++) {
            int curLeafID = ownerLeafOfData.get(i);
            if (tree.getLeaves().get(curLeafID).getError() <= maxError &&
                    tree.getLeaves().get(pairLeaf[curLeafID]).getError() <= maxError) continue;
            if (!bestCondLeaf[curLeafID].checkFeature(learn.data().row(i))) {
                ownerLeafOfData.set(i, pairLeaf[curLeafID]);
            }
        }

        return maxErr;
    }
}
