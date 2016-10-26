import java.util.List;

/**
 * Created by n_buga on 17.10.16.
 */
public class Leaf {
    private static int idCounter = 0;

    private int idLeaf = idCounter++;
    private ListFeatures listFeatures = new ListFeatures(idLeaf);;
    private double value;
    private double error;

    public Leaf() {
        error = -1;
    }

    public Leaf(double error) {
        this.error = error;
    }

    public Leaf(Leaf l) {
        listFeatures.addAllFeatures(l.listFeatures);
        value = l.value;
    }

    public double getError() {
        return error;
    }

    public void setError(double err) {
        error = err;
    }

    public int getID() {
        return  idLeaf;
    }

    public ListFeatures getListFeatures() {
        return listFeatures;
    }

    public void setValue(double value) {
        this.value = value;
    }

    public double getValue() {
        return value;
    }
}
