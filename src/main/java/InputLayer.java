import org.apache.commons.math3.linear.ArrayRealVector;

/**
 * Created by szale_000 on 2017-03-07.
 */
public class InputLayer extends Layer {

    public InputLayer(int numNeurons) {
        super(1, 4);
    }

    public void setInput(double[] data) {
        activationValues = new ArrayRealVector(data);
    }
}
