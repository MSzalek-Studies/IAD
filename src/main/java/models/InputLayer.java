package models;

import org.la4j.Matrix;
import utils.MatrixUtils;

/**
 * Created by szale_000 on 2017-03-07.
 */
public class InputLayer extends Layer {

    public InputLayer(int numNeurons, int numExamples) {
        this(numNeurons, numExamples, false);
    }

    public InputLayer(int numNeurons, int numExamples, boolean hasBias) {
        super(1, numNeurons, numExamples, hasBias);
    }

    public void setInput(Matrix data) {
        if (hasBias()) {
            activationValues = MatrixUtils.addBiasColumn(data);
        } else {
            activationValues = data;
        }
    }
}
