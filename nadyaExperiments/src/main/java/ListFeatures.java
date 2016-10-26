import com.spbsu.commons.math.vectors.Vec;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by n_buga on 17.10.16.
 */
public class ListFeatures {

    private List<Condition> listFeatures;
    int nodeId;

    public ListFeatures(int nodeId) {
        this.nodeId = nodeId;
        listFeatures = new ArrayList<Condition>();
    }

    public void addFeature(Condition t) {
        listFeatures.add(t);
    }

    public void addAllFeatures(ListFeatures t) {
        listFeatures.addAll(t.getList());
    }

    public List<Condition> getList() {
        return listFeatures;
    }

    public  boolean check(Vec x) {
        for (Condition condition: listFeatures) {
            if (!condition.checkFeature(x)) {
                return false;
            }
        }
        return true;
    }
}
