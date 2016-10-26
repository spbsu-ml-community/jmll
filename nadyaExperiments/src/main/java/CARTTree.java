import com.spbsu.commons.math.Func;
import com.spbsu.commons.math.vectors.Vec;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

/**
 * Created by n_buga on 17.10.16.
 */
public class CARTTree extends Func.Stub {
    private List<Leaf> leaves;

    public CARTTree() {
        leaves = new LinkedList<Leaf>();
    }

    public double value(Vec x) {
        for (Leaf leaf: leaves) {
            if (leaf.getListFeatures().check(x)) {
                return leaf.getValue();
            }
        }
        return 0;
    }

    public int dim() {
        return leaves.size();
    }

    public List<Leaf> getLeaves() {
        return leaves;
    }

    public void add(Leaf v) {
        leaves.add(v);
    }
}
