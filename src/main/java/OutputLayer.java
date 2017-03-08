import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import java.util.Arrays;

/**
 * Created by szale_000 on 2017-03-07.
 */
public class OutputLayer extends Layer {

    public OutputLayer(int numFeatures, int numNeurons) {
        super(numFeatures, numNeurons);
    }

    public double cost(RealVector expectedValues) {
        RealVector tempVec = activationValues.subtract(expectedValues);
        return Arrays.stream(tempVec.toArray()).map(d -> d*d).sum()/activationValues.getDimension();
    }

    public void calculateErrors(RealMatrix nextWeights) {
        RealVector expectedValues = nextWeights.getRowVector(0);
        calculateErrors(expectedValues);
    }

    public void calculateErrors(RealVector expectedValues) {
        errorOnValues = activationValues.subtract(expectedValues);
    }
}
