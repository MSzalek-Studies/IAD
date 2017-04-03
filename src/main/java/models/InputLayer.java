package models;

import org.la4j.Matrix;
import utils.MatrixUtils;

/**
 * Created by szale_000 on 2017-03-07.
 */
public class InputLayer extends Layer {

    public InputLayer(int numNeurons, int numExamples, double learningRate, double momentum) {
        this(numNeurons, numExamples, false, learningRate, momentum);
    }

    public InputLayer(int numNeurons, int numExamples, boolean hasBias, double learningRate, double momentum) {
        super(1, numNeurons, numExamples, hasBias, null, learningRate, momentum);
    }

    public void setInput(Matrix data) {
        activationValues = hasBias() ? MatrixUtils.addBiasColumn(data) : data;
    }
}
