package ml.data.impl;

/**
 * User: solar
 * Date: 26.12.10
 * Time: 19:09
 */
public class ChangedTarget extends DataSetImpl {
    final DataSetImpl parent;

    public ChangedTarget(DataSetImpl parent, double[] target) {
        super(parent.data(), target);
        this.parent = parent;
    }

    int[] order(int fIndex) {
        return parent.order(fIndex);
    }
}
